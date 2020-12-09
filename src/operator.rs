use num::traits::FloatConst;
use num::{Complex, Float, One, Zero};
use std::ops::{Index, IndexMut, Mul, MulAssign};

pub enum Operator<F: Float> {
    Single {
        target: u32,
        matrix: Matrix<F, 2>,
    },
    Controlled {
        target: u32,
        control: u32,
        matrix: Matrix<F, 4>,
    },
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Matrix<F: Float, const N: usize> {
    pub array: [[Complex<F>; N]; N],
}

impl<F: Float, const N: usize> Matrix<F, N> {
    pub fn new(array: [[Complex<F>; N]; N]) -> Self {
        Self { array }
    }

    pub fn zeros() -> Self {
        Self {
            array: [[Complex::<F>::zero(); N]; N],
        }
    }
}

impl<F: Float, const N: usize> Index<[usize; 2]> for Matrix<F, N> {
    type Output = Complex<F>;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.array[index[0]][index[1]]
    }
}

impl<F: Float, const N: usize> IndexMut<[usize; 2]> for Matrix<F, N> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.array[index[0]][index[1]]
    }
}

impl<F: Float, const N: usize> Mul<F> for Matrix<F, N> {
    type Output = Matrix<F, N>;
    fn mul(self, rhs: F) -> Self::Output {
        let mut array = self.array.clone();
        array
            .iter_mut()
            .flatten()
            .for_each(|x| *x = Complex::new(rhs * x.re, rhs * x.im));
        Matrix::new(array)
    }
}

impl<F: Float, const N: usize> MulAssign<F> for Matrix<F, N> {
    fn mul_assign(&mut self, rhs: F) {
        self.array
            .iter_mut()
            .flatten()
            .for_each(|x| *x = Complex::new(rhs * x.re, rhs * x.im));
    }
}

// Single-qubit gates
impl<F: Float + FloatConst> Matrix<F, 2> {
    pub fn identity() -> Self {
        let mut matrix = Self::zeros();
        matrix[[0, 0]] = Complex::one();
        matrix[[1, 1]] = Complex::one();
        matrix
    }

    pub fn x() -> Self {
        let mut matrix = Self::zeros();
        matrix[[0, 1]] = Complex::one();
        matrix[[1, 0]] = Complex::one();
        matrix
    }

    pub fn y() -> Self {
        let mut matrix = Self::zeros();
        matrix[[0, 1]] = -Complex::i();
        matrix[[1, 0]] = Complex::i();
        matrix
    }

    pub fn z() -> Self {
        let mut matrix = Self::zeros();
        matrix[[0, 0]] = Complex::one();
        matrix[[1, 1]] = -Complex::one();
        matrix
    }

    pub fn h() -> Self {
        let mut matrix = Self::zeros();
        matrix[[0, 0]] = Complex::one();
        matrix[[0, 1]] = Complex::one();
        matrix[[1, 0]] = Complex::one();
        matrix[[1, 1]] = -Complex::one();

        let norm = F::from(2).unwrap().recip().sqrt();
        matrix *= norm;

        matrix
    }

    pub fn phase(phase: F) -> Self {
        let mut matrix = Self::zeros();
        matrix[[0, 0]] = Complex::one();
        matrix[[1, 1]] = Complex::from_polar(F::one(), phase);
        matrix
    }

    pub fn s() -> Self {
        Self::phase(F::FRAC_PI_2())
    }

    pub fn s_dag() -> Self {
        Self::phase(-F::FRAC_PI_2())
    }

    pub fn t() -> Self {
        Self::phase(F::FRAC_PI_4())
    }

    pub fn t_dag() -> Self {
        Self::phase(-F::FRAC_PI_4())
    }
}

// Control qubit gates
impl<F: Float + FloatConst> Matrix<F, 4> {
    pub fn controlled(u: Matrix<F, 2>) -> Self {
        let mut matrix = Self::zeros();
        for i in 0..2 {
            matrix[[i, i]] = Complex::one();
            for j in 0..2 {
                matrix[[i + 2, j + 2]] = u[[i, j]];
            }
        }

        matrix
    }

    pub fn cnot() -> Self {
        Self::cx()
    }

    pub fn cx() -> Self {
        Self::controlled(Matrix::x())
    }

    pub fn cy() -> Self {
        Self::controlled(Matrix::y())
    }

    pub fn cz() -> Self {
        Self::controlled(Matrix::z())
    }

    pub fn swap() -> Self {
        let mut matrix = Self::zeros();
        matrix[[0, 0]] = Complex::one();
        matrix[[2, 1]] = Complex::one();
        matrix[[1, 2]] = Complex::one();
        matrix[[3, 3]] = Complex::one();

        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros() {
        let matrix = Matrix::<f32, 2>::zeros();
        for i in 0..1 {
            for j in 0..1 {
                assert_eq!(matrix[[i, j]], Complex::<f32>::new(0., 0.));
            }
        }
    }

    #[test]
    fn identity() {
        let matrix = Matrix::<f32, 2>::new([
            [Complex::one(), Complex::zero()],
            [Complex::zero(), Complex::one()],
        ]);
        assert_eq!(matrix, Matrix::identity());
    }

    #[test]
    fn x() {
        let matrix = Matrix::<f32, 2>::new([
            [Complex::zero(), Complex::one()],
            [Complex::one(), Complex::zero()],
        ]);
        assert_eq!(matrix, Matrix::x());
    }

    #[test]
    fn y() {
        let matrix = Matrix::<f32, 2>::new([
            [Complex::zero(), -Complex::i()],
            [Complex::i(), Complex::zero()],
        ]);
        assert_eq!(matrix, Matrix::y());
    }

    #[test]
    fn z() {
        let matrix = Matrix::<f32, 2>::new([
            [Complex::one(), Complex::zero()],
            [Complex::zero(), -Complex::one()],
        ]);
        assert_eq!(matrix, Matrix::z());
    }

    #[test]
    fn h() {
        let v = 2f32.recip().sqrt();
        let matrix = Matrix::<f32, 2>::new([
            [Complex::one(), Complex::one()],
            [Complex::one(), -Complex::one()],
        ]) * v;
        assert_eq!(matrix, Matrix::h());
    }

    #[test]
    fn cx() {
        let matrix = Matrix::<f32, 4>::new([
            [
                Complex::one(),
                Complex::zero(),
                Complex::zero(),
                Complex::zero(),
            ],
            [
                Complex::zero(),
                Complex::one(),
                Complex::zero(),
                Complex::zero(),
            ],
            [
                Complex::zero(),
                Complex::zero(),
                Complex::zero(),
                Complex::one(),
            ],
            [
                Complex::zero(),
                Complex::zero(),
                Complex::one(),
                Complex::zero(),
            ],
        ]);
        assert_eq!(matrix, Matrix::cx());
    }

    #[test]
    fn cy() {
        let matrix = Matrix::<f32, 4>::new([
            [
                Complex::one(),
                Complex::zero(),
                Complex::zero(),
                Complex::zero(),
            ],
            [
                Complex::zero(),
                Complex::one(),
                Complex::zero(),
                Complex::zero(),
            ],
            [
                Complex::zero(),
                Complex::zero(),
                Complex::zero(),
                -Complex::i(),
            ],
            [
                Complex::zero(),
                Complex::zero(),
                Complex::i(),
                Complex::zero(),
            ],
        ]);
        assert_eq!(matrix, Matrix::cy());
    }

    #[test]
    fn cz() {
        let matrix = Matrix::<f32, 4>::new([
            [
                Complex::one(),
                Complex::zero(),
                Complex::zero(),
                Complex::zero(),
            ],
            [
                Complex::zero(),
                Complex::one(),
                Complex::zero(),
                Complex::zero(),
            ],
            [
                Complex::zero(),
                Complex::zero(),
                Complex::one(),
                Complex::zero(),
            ],
            [
                Complex::zero(),
                Complex::zero(),
                Complex::zero(),
                -Complex::one(),
            ],
        ]);
        assert_eq!(matrix, Matrix::cz());
    }
}
