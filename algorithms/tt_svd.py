# algorithms/tt_svd.py

"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.

    Args:
        tensor:   DenseTensor с shape (n_0, n_1, ..., n_{d-1})
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    if tensor.ndim == 1:
        return TTTensor([tensor.reshape((1, tensor.shape[0], 1))])

    cores = []
    prev_r, C = 1, tensor.copy()
    frob_norm = tensor.norm()
    delta = eps * frob_norm / math.sqrt(tensor.ndim - 1) if frob_norm > 10 ** -30 else 0
    for k in range(tensor.ndim - 1):
        rows = prev_r * tensor.shape[k]
        C = C.reshape((rows, C.size // rows))
        U, singular_values, V = backend.svd(C)
        r = _compute_truncated_rank(singular_values, delta, max_rank)
        U = _truncate_columns(U, r, backend)
        cores.append(U.reshape((prev_r, tensor.shape[k], r)))
        singular_values = _truncate_vector(singular_values, r, backend)
        V = _truncate_rows(V, r, backend)
        C = _multiply_diag_matrix(singular_values, V, r, backend)
        prev_r = r
    cores.append(C.reshape((prev_r, tensor.shape[-1], 1)))
    return TTTensor(cores)

# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.

    Args:
        S:        DenseTensor (k,) — сингулярные значения по убыванию
        delta:    порог усечения
        max_rank: максимальный ранг (None = без ограничения)
    """
    if S.size == 0:
        return 1

    threshold = max(10 ** -12, abs(S.data[0]) * 10 ** -8)
    rank = 0
    for value in S.data:
        if abs(value) > threshold:
            rank += 1

    if rank == 0:
        rank = 1

    r = rank
    for curr_rank in range(1, rank + 1):
        tail = sum(S.data[i] ** 2 for i in range(curr_rank, rank))
        if tail <= delta ** 2:
            r = curr_rank
            break

    if max_rank is not None:
        r = min(r, max_rank)
    return max(1, r)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    rows, cols = matrix.shape
    data = []
    for i in range(rows):
        for j in range(rank):
            data.append(matrix.data[i * cols + j])
    return DenseTensor((rows, rank), data)


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
    cols = matrix.shape[1]
    return DenseTensor((rank, cols), matrix.data[:rank * cols])


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    return DenseTensor((rank,), vector.data[:rank])


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """

    return backend.matmul(backend.diag(diag_vec), matrix)
