use criterion::*;
use rand::*;
use simd_euclidean::*;

fn bench_random(c: &mut Criterion) {
    let mut rng = thread_rng();
    let input_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

    let mut group = c.benchmark_group("Random inputs f32");

    for &i in input_sizes.iter() {
        let mut a = Vec::with_capacity(i);
        let mut b = Vec::with_capacity(i);

        for _ in 0..i {
            a.push(rng.gen::<f32>());
        }
        for _ in 0..i {
            b.push(rng.gen::<f32>());
        }

        group.throughput(Throughput::Elements(i as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Naive_{i}")),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher
                    .iter(|| Naive::distance_z_normalized(black_box(*a), black_box(*b), 0.0, 1.0));
            },
        );
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Vectorized_{i}")),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    Vectorized::distance_z_normalized(black_box(*a), black_box(*b), 0.0, 1.0)
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_random);
criterion_main!(benches);
