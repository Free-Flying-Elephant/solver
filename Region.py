

import matplotlib.pyplot as plt
import numpy as np

import func as fc
import iofc

from Cell import Cell
from Node import Node


class Region:

    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.cells: list[Cell] = []
        self.err_seq: list[float] = []
        self.err_lst: list[list[float]] = []
        self.err: np.ndarray = np.ones((1), dtype=np.float64)
        self.errm: np.ndarray = np.ones((1), dtype=np.float64)
        self.errx: np.ndarray = np.ones((1), dtype=np.float64)
        self.erry: np.ndarray = np.ones((1), dtype=np.float64)
        self.errh: np.ndarray = np.ones((1), dtype=np.float64)
        self.A: np.ndarray = np.zeros((2, 2), dtype=np.float64) # heat convection in solid
        self.Ax: np.ndarray = np.zeros((2, 2), dtype=np.float64) # x momentum ico
        self.Ay: np.ndarray = np.zeros((2, 2), dtype=np.float64) # y momentum ico
        self.Am: np.ndarray = np.zeros((2, 2), dtype=np.float64) # mass conservation ico
        self.Ah: np.ndarray = np.zeros((2, 2), dtype=np.float64) # energy conservation ico
        self.b: np.ndarray = np.zeros((2, 1), dtype=np.float64)
        self.bx: np.ndarray = np.zeros((2, 1), dtype=np.float64)
        self.by: np.ndarray = np.zeros((2, 1), dtype=np.float64)
        self.bm: np.ndarray = np.zeros((2, 1), dtype=np.float64)
        self.bh: np.ndarray = np.zeros((2, 1), dtype=np.float64)
        self.xm: np.ndarray = np.zeros((2, 1), dtype=np.float64)
        self.xx: np.ndarray = np.zeros((2, 1), dtype=np.float64)
        self.xy: np.ndarray = np.zeros((2, 1), dtype=np.float64)
        self.xh: np.ndarray = np.zeros((2, 1), dtype=np.float64)

    def plot(self) -> None:
        palette: list[str] = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple", "tab:pink", "tab:brown"] * 10
        plt.figure("error")
        plt.plot(self.err_seq)
        plt.yscale("log")
        plt.xlim(0, len(self.err_seq)-1)
        plt.grid(True)
        plt.figure("mesh")
        for node in self.nodes:
            plt.plot(node.x, node.y, marker="o")
            plt.annotate(str(node.id), tuple(node.coord))
        k = -1
        for cell in self.cells:
            k += 1
            plt.plot(cell.centr[0, 0], cell.centr[1, 0], marker="d")
            plt.annotate(f"{cell.T:.1f}", tuple(cell.centr))
            for edge in cell.edge:
                plt.plot([edge.N1.x, edge.N2.x], [edge.N1.y, edge.N2.y])
                plt.plot(edge.C[0, 0], edge.C[1, 0], marker="*")
                plt.plot([edge.C[0, 0], edge.C[0, 0] + edge.unit_normal[0, 0] / 10], [edge.C[1, 0], edge.C[1, 0] + edge.unit_normal[1, 0] / 10], color=palette[k])
        plt.axis("equal")
        plt.show()

    def assign_neighbours(self, bc_cells: list[Cell]) -> None:

        for i, cell in enumerate(self.cells):
            for j, edge in enumerate(cell.edge):
                new_edge = False
                for i_n, cell_n in enumerate(self.cells):
                    for j_n, edge_n in enumerate(cell_n.edge):
                        if edge_n.N1 is edge.N2 and edge_n.N2 is edge.N1:
                            cell.neighbour.append(cell_n)
                            cell.neighbour_id.append(i_n)
                            new_edge = True
                            break
                    if new_edge is True: break
                else:
                    # boundary
                    cell.neighbour.append(bc_cells[edge.bc_id - 1])
                    cell.neighbour_id.append(-1)
            if len(cell.neighbour_id) != 3: raise RuntimeWarning
        return

    def iterate_temp_solid(self, tol: float, max_iter: int) -> bool:

        self.err[0] = 1.
        self.x = np.zeros(np.shape(self.b), dtype=np.float64)
        for i, cell in enumerate(self.cells):
            self.x[i, 0] = self.cells[i].T

        for iter in range(max_iter):
            if self.err[0] <= tol: break

            for i, cell in enumerate(self.cells):
                self.A[i, i] = 0.
                self.b[i, 0] = 0.
                for j, (edge, idx_n) in enumerate(zip(cell.edge, cell.neighbour_id)):
                    if idx_n == -1:
                        D: float = cell.neighbour[j].k * edge.L / (3 ** .5 / 2) * 2 # diffusion coefficient
                        self.b[i, 0] = self.b[i, 0] + D * cell.neighbour[j].T
                    else:
                        D: float = cell.neighbour[j].k * edge.L / (3 ** .5 / 2) # diffusion coefficient
                        self.A[i, idx_n] = -D # left side coefficient
                    self.A[i, i] = self.A[i, i] + D # add to diagonal
            
            fc.gauss_seidel_step(self.A, self.b, self.x, self.err)
            self.err_seq.append(self.err[0])
            
            for i, cell in enumerate(self.cells):
                self.cells[i].T = self.x[i, 0]

            print(f"[{iter+1}] : {self.err[0]:.3e} [K]")

        else: return False
        return True

    def iterate_ico(self, tol: float, max_iter: int) -> bool:
        self.err = np.ones((4), dtype=np.float64)
        for i in range(4):
            self.err_lst.append([])
        self.xm = np.ones(np.shape(self.bm), dtype=np.float64)
        self.xx = np.ones(np.shape(self.bx), dtype=np.float64)
        self.xy = np.ones(np.shape(self.by), dtype=np.float64)
        self.xh = np.ones(np.shape(self.bh), dtype=np.float64)

        for iter in range(max_iter):
            if np.max(self.err) <= tol: break

            for i, cell in enumerate(self.cells):
                self.Am[i, i] = 0.
                self.Ax[i, i] = 0.
                self.Ay[i, i] = 0.
                self.Ah[i, i] = 0.
                self.bm[i, 0] = 0.
                self.bx[i, 0] = 0.
                self.by[i, 0] = 0.
                self.bh[i, 0] = 0.
                for j, (edge, idx_n) in enumerate(zip(cell.edge, cell.neighbour_id)):

                    dT: np.ndarray = np.asarray([(cell.T - cell.neighbour[j].T) / (cell.centr - cell.neighbour[j].centr)], dtype=np.float64)

                    F: float = (cell.rho + cell.neighbour[j].rho) / 2 * np.dot(edge.unit_normal.flatten(), edge.U.flatten()) * edge.L # mass conservation
                    dp: np.ndarray = -(cell.p + cell.neighbour[j].p) / 2 * edge.unit_normal * edge.L # p * n_ * L
                    dpe: float = -(cell.p + cell.neighbour[j].p) / 2 * np.dot(edge.unit_normal.flatten(), edge.U.flatten()) * edge.L
                    dte: float = cell.k * np.dot(edge.unit_normal.flatten(), dT) * edge.L

                    if idx_n == -1:
                        self.bx[i, 0] = F + dp[0, 0]
                        self.by[i, 0] = F + dp[1, 0]
                        self.bh[i, 0] = F + dpe + dte
                        pass # TODO

                    else:
                        self.Am[i, idx_n] = -F
                        self.Ax[i, idx_n] = -F
                        self.Ay[i, idx_n] = -F
                        self.Ah[i, idx_n] = -F

                    self.Am[i, i] = self.Am[i, i] + F
                    self.Ax[i, i] = self.Ax[i, i] + F
                    self.Ay[i, i] = self.Ay[i, i] + F
                    self.Ah[i, i] = self.Ah[i, i] + F
                    
                    self.bx[i, 0] = self.bx[i, 0] + dp[0, 0]
                    self.by[i, 0] = self.by[i, 0] + dp[1, 0]
                    self.bh[i, 0] = self.bh[i, 0] + dpe + dte
            
            fc.gauss_seidel_step(self.Am, self.bm, self.xm, self.errm)
            self.err_lst[0].append(self.errm[0])
            fc.gauss_seidel_step(self.Ax, self.bx, self.xx, self.errx)
            self.err_lst[1].append(self.errx[0])
            fc.gauss_seidel_step(self.Ay, self.by, self.xy, self.erry)
            self.err_lst[2].append(self.erry[0])
            fc.gauss_seidel_step(self.Ah, self.bh, self.xh, self.errh)
            self.err_lst[3].append(self.errh[0])

            self.err = np.asarray([self.errm[0], self.errx[0], self.erry[0], self.errh[0]])
            
            for i, cell in enumerate(self.cells):
                cell.rho = self.bm[i, 0]
                cell.T = self.xh[i, 0] / (cell.cp - cell.R) / cell.rho
                cell.p = cell.rho * cell.T * cell.R
                for j, (edge, idx_n) in enumerate(zip(cell.edge, cell.neighbour_id)):
                    if idx_n == -1:
                        self.cells[i].edge[j].U[0, 0] = 0.
                        self.cells[i].edge[j].U[1, 0] = 0.
                        pass # TODO
                    else:
                        self.cells[i].edge[j].U[0, 0] = (self.xx[i, 0] / self.xm[i, 0] + self.xx[idx_n, 0] / self.xm[idx_n, 0]) / 2
                        self.cells[i].edge[j].U[1, 0] = (self.xy[i, 0] / self.xm[i, 0] + self.xy[idx_n, 0] / self.xm[idx_n, 0]) / 2

            print(f"[{iter+1}] : {self.err[0]:.3e} [K]")

        else: return False
        return True
            

def main() -> None:

    # initialize mesh region (zone)
    mesh = Region()

    # read mesh node positions from file
    msh: dict[str, list[float]] = iofc.read_mesh(r"mesh\\coord.json")
    X: list[float] = msh["x"]
    Y: list[float] = msh["y"]

    # initialize nodes and assing coordinates read form file
    for i, (x, y) in enumerate(zip(X, Y)):
        mesh.nodes.append(Node(i, x, y))

    # read vertex indeces
    idx: list[list[int]] = iofc.read_idx(r"mesh\\idx.json")

    for i in range(len(idx)):
        p: list[Node] = [mesh.nodes[k] for k in idx[i]]
        mesh.cells.append(Cell(p))
    
    # update matrix
    mesh.A = np.zeros((len(mesh.cells), len(mesh.cells)), dtype=np.float64)
    mesh.b = np.zeros((len(mesh.cells), 1), dtype=np.float64)

    mesh.Am = np.zeros((len(mesh.cells), len(mesh.cells)), dtype=np.float64)
    mesh.bm = np.zeros((len(mesh.cells), 1), dtype=np.float64)
    mesh.Ax = np.zeros((len(mesh.cells), len(mesh.cells)), dtype=np.float64)
    mesh.bx = np.zeros((len(mesh.cells), 1), dtype=np.float64)
    mesh.Ay = np.zeros((len(mesh.cells), len(mesh.cells)), dtype=np.float64)
    mesh.by = np.zeros((len(mesh.cells), 1), dtype=np.float64)
    mesh.Ah = np.zeros((len(mesh.cells), len(mesh.cells)), dtype=np.float64)
    mesh.bh = np.zeros((len(mesh.cells), 1), dtype=np.float64)

    # read boundary conditions from file
    case: dict[str, list[int]] = iofc.read_bc(r"case\\bc.json")
    ed: list[int] = case["edge"] # edge identifier (which edge is on the boundary; -1 means no bc on the cell boundary)
    bc: list[int] = case["bc"] # bc identifier (which BC is on that edge)
    for i, cell in enumerate(mesh.cells):
        if ed[i] < 0: continue
        cell.edge[ed[i]].bc_id = bc[i]
    
    # mesh.cells[0].edge[0].bc_id = 1 # no-slip / no-heat-flux wall
    # mesh.cells[1].edge[2].bc_id = 2 # fluid inlet / heat source
    # mesh.cells[2].edge[0].bc_id = 3 # fluid outlet / heat sink

    # create all boundary cells
    bc_cells: list[Cell] = []
    var_dct_cell_lst: list[dict[str, float | int]] = []
    for i in range(3):
        bc_cells.append(Cell([]))
        var_dct_cell_lst.append(fc.cell_inp_var_init(i))
    
    # assign the values to boundary cells
    for i in range(3):
        if iofc.read_inp_file(rf"case\\{i}.inp", var_dct_cell_lst[i]) != 0: raise NameError
        slots = tuple(bc_cells[i].__slots__)
        for key, val in var_dct_cell_lst[i].items():
            if np.isnan(val): continue
            if key in slots: bc_cells[i].__setattr__(key, val)
            else:
                if key == "u": bc_cells[i].U[0, 0] = val
                elif key == "v": bc_cells[i].U[1, 0] = val
                else: raise KeyError
    
    # assign boundary cell to cell inside region of interest
    mesh.assign_neighbours(bc_cells)
    
    # compute the solution + plot the solution
    conv: bool = mesh.iterate_temp_solid(1e-3, 100)
    # conv: bool = mesh.iterate_ico(1e-3, 100)
    print(f"Converged: {conv}")
    mesh.plot()



if __name__ == "__main__":
    main()
