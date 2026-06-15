# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [core.copy() for core in tt.cores]

    for k in range(tt.order - 1):
        r1, n, r2 = cores[k].shape
        matrix = cores[k].reshape((r1 * n, r2))
        U, S, V = backend.svd(matrix)
        rank = _numerical_rank(S)

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


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [core.copy() for core in tt.cores]

    for k in range(tt.order - 1, 0, -1):
        r1, n, r2 = cores[k].shape
        matrix = cores[k].reshape((r1, n * r2))
        U, S, V = backend.svd(matrix)
        rank = _numerical_rank(S)

        U = _truncate_columns(U, rank, backend)
        S = _truncate_vector(S, rank, backend)
        V = _truncate_rows(V, rank, backend)

        cores[k] = V.reshape((rank, n, r2))
        transfer = _multiply_columns_by_diag(U, S, backend)

        prev_core = cores[k - 1]
        prev_matrix = prev_core.reshape((prev_core.shape[0] * prev_core.shape[1], r1))
        cores[k - 1] = backend.matmul(prev_matrix, transfer).reshape(
            (prev_core.shape[0], prev_core.shape[1], rank)
        )

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг матрицы по вектору сингулярных значений.

    Сингулярное число sigma_i считаем ненулевым, если:
        |sigma_i| > max(abs_tol, rel_tol * max(sigma_1, ..., sigma_n))

    Args:
        S:       одномерный тензор формы (k,) — сингулярные значения
                 в порядке убывания
        rel_tol: относительный допуск (по умолчанию 1e-8)
        abs_tol: абсолютный допуск (по умолчанию 1e-12)
    """
    if S.size == 0:
        return 1

    threshold = max(abs_tol, rel_tol * max(abs(x) for x in S.data))
    rank = 0
    for value in S.data:
        if abs(value) > threshold:
            rank += 1
    return max(1, rank)


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
        rank:     длина диагонального вектора
        backend:  интерфейс backend
    """
    return backend.matmul(backend.diag(diag_vec), matrix)


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает результат произведения обычной матрицы на диагональную:
        matrix @ diag(diag_vec)

    Args:
        matrix:   двумерный тензор формы (m, n)
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        backend:  интерфейс backend
    """
    rows, cols = matrix.shape
    data = []
    for i in range(rows):
        for j in range(cols):
            data.append(matrix.data[i * cols + j] * diag_vec.data[j])
    return DenseTensor((rows, cols), data)
