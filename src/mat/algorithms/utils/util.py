import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

# Row-normalize to do mean over neighbors, but keep isolated rows as identity rows
def normalize_dense(M):
    row_sum = M.sum(dim=1, keepdim=True)  # [N,1]
    W = M / torch.clamp(row_sum, min=1)
    isolated = (row_sum.squeeze(1) == 0)
    if isolated.any():
        W[isolated] = 0
        W[isolated, isolated] = 1
    return W

def normalize_sparse(S):
    row_sum = torch.sparse.sum(S, dim=1).to_dense()  # [N]
    S = S.coalesce()
    idx = S.indices()   # [2, nnz]
    vals = S.values()   # [nnz]
    denom = torch.clamp(row_sum[idx[0]], min=1)
    nvals = vals / denom
    W = torch.sparse_coo_tensor(idx, nvals, S.shape, device=S.device, dtype=S.dtype).coalesce()
    # Add identity for isolated rows
    isolated = (row_sum == 0).nonzero(as_tuple=False).view(-1)
    if isolated.numel() > 0:
        eye_idx = isolated
        eye = torch.sparse_coo_tensor(
            torch.stack([eye_idx, eye_idx]),
            torch.ones_like(eye_idx, dtype=S.dtype, device=S.device),
            S.shape, device=S.device, dtype=S.dtype
        ).coalesce()
        W = (W + eye).coalesce()
    return W


def average_agent_encoders_by_adj(agent_encoders, A, include_buffers=False, add_self_loops=True):
    """
    Vectorized parameter averaging over neighbors defined by adjacency A.

    Args:
        agent_encoders (nn.ModuleList): list of N encoder modules with identical structure
        A (torch.Tensor or np.ndarray or torch.sparse): adjacency [N, N].
            Values are interpreted as weights; >0 means neighbor.
        include_buffers (bool): if True, also average buffers (e.g., running_mean/var).
        add_self_loops (bool): if True, adds self-loops before normalization.
    """
    N = len(agent_encoders)
    if N == 0:
        return

    # Choose a reference device/dtype from the first param
    ref_param = next(agent_encoders[0].parameters(), None)
    ref_device = ref_param.device if ref_param is not None else torch.device("cpu")
    ref_dtype = ref_param.dtype if ref_param is not None else torch.float32

    # Prepare adjacency on the right device/dtype
    if not torch.is_tensor(A):
        A = torch.tensor(A, dtype=ref_dtype, device=ref_device)
    else:
        A = A.to(device=ref_device, dtype=ref_dtype)

    # Optionally add self-loops
    if add_self_loops and not A.is_sparse:
        A = A.clone()
        A = A + torch.eye(A.size(0), device=A.device, dtype=A.dtype)

    # Build a row-normalized weight matrix W with isolated-row handling

    use_sparse = A.is_sparse
    W = normalize_sparse(A) if use_sparse else normalize_dense(A)

    # Helper: iterate parameters (and optionally buffers) by name across agents
    def iter_named_tensors(modules, include_buffers=False):
        # Yields (name, list_of_tensors_across_agents, is_param)
        # Ensures consistent ordering across identical module structures.
        names = []
        ref = modules[0]
        if include_buffers:
            # State dict includes both params and buffers
            for name, _ in ref.state_dict().items():
                names.append(name)
        else:
            for name, _ in ref.named_parameters():
                names.append(name)

        for name in names:
            tensors = []
            for m in modules:
                if include_buffers:
                    t = dict(m.named_parameters()).get(name, None)
                    if t is None:
                        # Then it must be a buffer
                        t = dict(m.named_buffers())[name]
                else:
                    t = dict(m.named_parameters())[name]
                tensors.append(t)
            yield name, tensors, (not include_buffers) or (name in dict(modules[0].named_parameters()))

    with torch.no_grad():
        # For each (param or buffer) name, stack across agents and average via W
        for name, tensors, is_param in iter_named_tensors(agent_encoders, include_buffers=include_buffers):
            # Expect identical shapes per agent
            shape = tensors[0].shape
            # Some buffers might be non-float (e.g., integer). Skip non-float unless you want special handling.
            if not torch.is_floating_point(tensors[0]):
                continue

            # Stack over agents: [N, *shape]
            P = torch.stack([t.to(ref_device) for t in tensors], dim=0)  # [N, ...]
            # Weighted neighbor average: newP[i,...] = sum_j W[i,j] * P[j,...]
            # Dense or sparse path:
            if use_sparse:
                # Flatten, sparse mm, then reshape
                P_flat = P.reshape(N, -1)                     # [N, D]
                new_flat = torch.sparse.mm(W, P_flat)         # [N, D]
                newP = new_flat.reshape_as(P)                 # [N, *shape]
            else:
                # Einsum keeps original shape without flattening
                newP = torch.einsum('ij,j...->i...', W, P)    # [N, *shape]

            # Copy back
            for i in range(N):
                # Respect requires_grad for parameters; buffers don't have requires_grad
                if is_param:
                    if tensors[i].requires_grad:
                        tensors[i].data.copy_(newP[i])
                else:
                    tensors[i].copy_(newP[i])


def average_attention_params_by_adj(atts, A, add_self_loops=True):
    """
    Vectorized averaging of attention parameters across neighbors.

    Args:
        atts (nn.ModuleList): nested like atts[k][i] where
                              k = hop index (0..H-1), i = agent index (0..N-1),
                              each atts[k][i] is an nn.Parameter (same shape across i for a given k).
        A (Tensor or np.ndarray or sparse): adjacency [N, N]; A[i,j] > 0 => j is neighbor of i
        add_self_loops (bool): if True, treat each node as its own neighbor as well.
    Behavior:
        - Unweighted mean over neighbors where A[i,j] > 0.
        - Isolated nodes (no neighbors) keep their own params.
    """
    H = len(atts)
    if H == 0:
        return

    # Infer N, param device/dtype from first entry
    N = len(atts[0])
    ref_param = atts[0][0]
    ref_device = ref_param.device
    ref_dtype = ref_param.dtype

    # Prepare adjacency on right device/dtype
    if not torch.is_tensor(A):
        A = torch.tensor(A, dtype=ref_dtype, device=ref_device)
    else:
        A = A.to(device=ref_device, dtype=ref_dtype)

    # # Convert to binary mask (>0), optional self-loops
    # if A.is_sparse:
    #     A = A.coalesce()
    #     idx = A.indices()
    #     vals = (A.values() > 0).to(ref_dtype)
    #     B = torch.sparse_coo_tensor(idx, vals, A.shape, device=A.device, dtype=ref_dtype).coalesce()
    #     if add_self_loops:
    #         eye_idx = torch.arange(N, device=A.device)
    #         eye = torch.sparse_coo_tensor(
    #             torch.stack([eye_idx, eye_idx]), torch.ones(N, dtype=ref_dtype, device=A.device),
    #             (N, N), device=A.device, dtype=ref_dtype
    #         )
    #         B = (B + eye).coalesce()
    # else:
    #     B = (A > 0).to(dtype=ref_dtype)
    #     if add_self_loops:
    #         B = B.clone()
    #         B.fill_diagonal_(1)

    # Optionally add self-loops
    if add_self_loops and not A.is_sparse:
        A = A.clone()
        B = A + torch.eye(A.size(0), device=A.device, dtype=A.dtype)

    use_sparse = B.is_sparse
    W = normalize_sparse(B) if use_sparse else normalize_dense(B)

    with torch.no_grad():
        # Stack parameters: for each hop k, stack across agents -> P[k] = [N, *shape_k]
        # If all hops have same shape, we can stack all at once; otherwise handle each k independently.
        same_shape = all(tuple(atts[k][0].shape) == tuple(atts[0][0].shape) for k in range(H))

        if (not use_sparse) and same_shape:
            # Dense, all hops share same shape: do a single einsum for all hops
            shape = atts[0][0].shape
            P = torch.stack([torch.stack([atts[k][i] for i in range(N)], dim=0)
                             for k in range(H)], dim=0)       # [H, N, *shape]
            # Weighted neighbor mean: new[k,i,...] = sum_j W[i,j]*P[k,j,...]
            newP = torch.einsum('ij,hj...->hi...', W, P)       # [H, N, *shape]
            # Copy back
            for k in range(H):
                for i in range(N):
                    atts[k][i].data.copy_(newP[k, i])

        else:
            # Fallback per-hop (still vectorized across agents for each hop)
            for k in range(H):
                shape = atts[k][0].shape
                P = torch.stack([atts[k][i] for i in range(N)], dim=0)  # [N, *shape]

                if use_sparse:
                    P_flat = P.reshape(N, -1)                            # [N, D]
                    new_flat = torch.sparse.mm(W, P_flat)                # [N, D]
                    newP = new_flat.reshape_as(P)                        # [N, *shape]
                else:
                    newP = torch.einsum('ij,j...->i...', W, P)           # [N, *shape]

                for i in range(N):
                    atts[k][i].data.copy_(newP[i])
