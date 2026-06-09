import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class LightGCNConv(MessagePassing):
    """LightGCN-style normalized neighbor aggregation used by MSCRS semantic graphs."""

    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out


CustomGCNConv = LightGCNConv
