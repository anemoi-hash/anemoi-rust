use criterion::{black_box, criterion_group, criterion_main, Criterion};

extern crate anemoi;
use anemoi::ed_on_bls12_377::*;
use anemoi::{Jive, Sponge};
use rand_core::OsRng;
use rand_core::RngCore;

use ark_ff::One;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function(
        "anemoi-jive/ed_on_bls12_377/2-1 (128 bits security) - 2-to-1 compression",
        |bench| {
            let v = [Felt::one(); anemoi_2_1::STATE_WIDTH];

            bench.iter(|| anemoi_2_1::AnemoiEdOnBls12_377_2_1::compress(black_box(&v)))
        },
    );

    c.bench_function(
        "anemoi-jive/ed_on_bls12_377/4-3 (128 bits security) - 2-to-1 compression",
        |bench| {
            let v = [Felt::one(); anemoi_4_3::STATE_WIDTH];

            bench.iter(|| anemoi_4_3::AnemoiEdOnBls12_377_4_3::compress(black_box(&v)))
        },
    );

    c.bench_function(
        "anemoi-jive/ed_on_bls12_377/4-3 (128 bits security) - 4-to-1 compression",
        |bench| {
            let v = [Felt::one(); anemoi_4_3::STATE_WIDTH];

            bench.iter(|| anemoi_4_3::AnemoiEdOnBls12_377_4_3::compress_k(black_box(&v), 4))
        },
    );

    c.bench_function(
        "anemoi-sponge/ed_on_bls12_377/2-1 (128 bits security) - hash 10KB",
        |bench| {
            let mut data = vec![0u8; 10 * 1024];
            let mut rng = OsRng;
            rng.fill_bytes(&mut data);

            bench.iter(|| anemoi_2_1::AnemoiEdOnBls12_377_2_1::hash(black_box(&data)))
        },
    );

    c.bench_function(
        "anemoi-sponge/ed_on_bls12_377/4-3 (128 bits security) - hash 10KB",
        |bench| {
            let mut data = vec![0u8; 10 * 1024];
            let mut rng = OsRng;
            rng.fill_bytes(&mut data);

            bench.iter(|| anemoi_4_3::AnemoiEdOnBls12_377_4_3::hash(black_box(&data)))
        },
    );
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = criterion_benchmark);
criterion_main!(benches);
