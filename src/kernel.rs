use crate::operator::Matrix;
use itertools::Itertools;
use num::{Complex, Float};
use rayon::iter::ParallelIterator;
use rayon::prelude::*;

#[inline(never)]
pub fn single_serial<F: Float>(k: u32, matrix: Matrix<F, 2>, state: &mut [Complex<F>]) {
    for chunk in state.chunks_exact_mut(2usize.pow(k + 1)) {
        let (in0, in1) = chunk.split_at_mut(2usize.pow(k));
        for (a0, a1) in in0.iter_mut().zip(in1.iter_mut()) {
            let tmp0 = *a0;
            let tmp1 = *a1;

            *a0 = matrix[[0, 0]] * tmp0 + matrix[[0, 1]] * tmp1;
            *a1 = matrix[[1, 0]] * tmp1 + matrix[[1, 1]] * tmp1;
        }
    }
}

#[inline(never)]
pub fn single_par_outer<F: Float + Send + Sync>(
    k: u32,
    matrix: Matrix<F, 2>,
    state: &mut [Complex<F>],
) {
    state
        .par_chunks_exact_mut(2usize.pow(k + 1))
        .for_each(|chunk| {
            let (in0, in1) = chunk.split_at_mut(2usize.pow(k));
            for (a0, a1) in in0.iter_mut().zip(in1.iter_mut()) {
                let tmp0 = *a0;
                let tmp1 = *a1;

                *a0 = matrix[[0, 0]] * tmp0 + matrix[[0, 1]] * tmp1;
                *a1 = matrix[[1, 0]] * tmp1 + matrix[[1, 1]] * tmp1;
            }
        });
}

#[inline(never)]
pub fn single_par_inner<F: Float + Send + Sync, const TILE: usize>(
    k: u32,
    matrix: Matrix<F, 2>,
    state: &mut [Complex<F>],
) {
    for chunk in state.chunks_exact_mut(2usize.pow(k + 1)) {
        let (in0, in1) = chunk.split_at_mut(2usize.pow(k));
        in0.par_chunks_exact_mut(TILE)
            .zip(in1.par_chunks_exact_mut(TILE))
            .for_each(|(chunk0, chunk1)| {
                for (a0, a1) in chunk0.iter_mut().zip(chunk1.iter_mut()) {
                    let tmp0 = *a0;
                    let tmp1 = *a1;

                    *a0 = matrix[[0, 0]] * tmp0 + matrix[[0, 1]] * tmp1;
                    *a1 = matrix[[1, 0]] * tmp1 + matrix[[1, 1]] * tmp1;
                }
            });
    }
}
