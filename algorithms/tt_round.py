# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize
from algorithms.tensor_operations import tt_norm


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами

    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    if tt.order == 1:
        return tt.copy()

    rounded = right_canonicalize(tt, backend)
    cores = [core.copy() for core in rounded.cores]
    frob_norm = tt_norm(tt, backend)
    delta = eps * frob_norm / math.sqrt(tt.order - 1) if frob_norm > 10 ** -30 else 0

    for k in range(tt.order - 1):
        r1, n, r2 = cores[k].shape
        matrix = cores[k].reshape((r1 * n, r2))
        U, S, V = backend.svd(matrix)
        rank = _compute_rank(S, delta, max_rank)

        U = _truncate_columns(U, rank, backend)
        S = _truncate_vector(S, rank, backend)
        V = _truncate_rows(V, rank, backend)

        cores[k] = U.reshape((r1, n, rank))
        transfer = _multiply_diag_matrix(S, V, rank, backend)

        next_core = cores[k + 1]
        next_matrix = next_core.reshape((r2, next_core.shape[1] * next_core.shape[2]))
        cores[k + 1] = backend.matmul(transfer, next_matrix).reshape(
            (rank, next_core.shape[1], next_core.shape[2])
        )

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимально допустимый ранг (None = без ограничения)
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
