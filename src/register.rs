use crate::kernel;
use crate::operator::*;
use num::{Complex, Float, Zero};
use smallvec::SmallVec;

pub struct Register<F: Float> {
    num_qubits: u32,
    state: Vec<Complex<F>>,
    tmp: Vec<Complex<F>>,
}

impl<F: Float> Register<F> {
    pub fn new(num_qubits: u32) -> Self {
        if num_qubits == 0 {
            panic!("Number of qubits cannot be 0!");
        }
        let state = vec![Complex::zero(); 2usize.pow(num_qubits)];
        let tmp = vec![Complex::zero(); 2usize.pow(num_qubits - 1)];
        Self {
            num_qubits,
            state,
            tmp,
        }
    }
}

impl<F: Float + Send + Sync> Register<F> {
    pub fn apply(&mut self, gate: Operator<F>) {
        match gate {
            Operator::Single { target, matrix } => {
                kernel::single_par_outer(target, matrix, &mut self.state);
            }
            Operator::Controlled {
                target,
                control,
                matrix,
            } => {
                self.apply_controlled(target, control, matrix);
            }
        }
    }

    fn apply_controlled(&mut self, target: u32, control: u32, matrix: Matrix<F, 4>) {
        let n = self.num_qubits as usize;
        let dh = 2usize.pow(control);
        let dl = 2usize.pow(target);
        let invert = target > control;

        for i in (0..n).step_by(2 * if invert { dl } else { dh }) {
            for j in (0..if invert { dl } else { dh }).step_by(2 * if invert { dh } else { dl }) {
                for k in 0..if invert { dh } else { dl } {
                    let tmp = i + j + k;
                    let idx = [tmp, tmp + dl, tmp + dh, tmp + dh + dl];

                    let tmp: SmallVec<[Complex<F>; 4]> =
                        (0..4).map(|i| self.state[idx[i]]).collect();

                    for t in 0..4 {
                        self.state[idx[t]] = (0..4).map(|s| matrix[[t, s]] * tmp[idx[s]]).sum();
                    }
                }
            }
        }
    }
}
