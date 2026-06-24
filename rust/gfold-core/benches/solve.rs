use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use gfold_core::config::Config;
use gfold_core::assemble::assemble;
use gfold_core::solve::solve;

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("gfold");
    for &n in &[20usize, 50, 100, 200] {
        let mut cfg = Config::default();
        cfg.solver.n = n;
        group.bench_with_input(BenchmarkId::new("assemble", n), &cfg, |b, cfg| {
            b.iter(|| assemble(cfg));
        });
        group.bench_with_input(BenchmarkId::new("solve", n), &cfg, |b, cfg| {
            b.iter(|| solve(cfg).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
