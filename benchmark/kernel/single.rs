use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use iqs::kernel;
use iqs::operator::Matrix;
use num::{Complex, Zero};

const N: u32 = 14;

fn single(c: &mut Criterion) {
    let mut state = vec![Complex::<f32>::zero(); 2usize.pow(N)];
    let matrix = Matrix::<f32, 2>::h();

    let mut group = c.benchmark_group("single");
    for k in 0..N {
        group.bench_with_input(BenchmarkId::new("serial", k), &k, |b, k| {
            b.iter(|| {
                kernel::single_serial(black_box(*k), black_box(matrix), black_box(&mut state))
            })
        });
        group.bench_with_input(BenchmarkId::new("par_outer", k), &k, |b, k| {
            b.iter(|| {
                kernel::single_par_outer(black_box(*k), black_box(matrix), black_box(&mut state))
            })
        });
        group.bench_with_input(BenchmarkId::new("par_inner", k), &k, |b, k| {
            b.iter(|| {
                kernel::single_par_inner::<f32, 16>(
                    black_box(*k),
                    black_box(matrix),
                    black_box(&mut state),
                )
            })
        });
    }
}

criterion_group!(benches, single);
criterion_main!(benches);
