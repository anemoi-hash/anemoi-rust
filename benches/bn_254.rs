use criterion::{black_box, criterion_group, criterion_main, Criterion};

extern crate anemoi;
use anemoi::bn_254::*;
use anemoi::{Jive, Sponge};
use rand_core::OsRng;
use rand_core::RngCore;

use ark_ff::One;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function(
        "anemoi-jive/bn_254/2-1 (128 bits security) - 2-to-1 compression",
        |bench| {
            let v = [Felt::one(); anemoi_2_1::STATE_WIDTH];

            bench.iter(|| anemoi_2_1::AnemoiHash::compress(black_box(&v)))
        },
    );

    c.bench_function(
        "anemoi-jive/bn_254/4-3 (128 bits security) - 2-to-1 compression",
        |bench| {
            let v = [Felt::one(); anemoi_4_3::STATE_WIDTH];

            bench.iter(|| anemoi_4_3::AnemoiHash::compress(black_box(&v)))
        },
    );

    c.bench_function(
        "anemoi-jive/bn_254/4-3 (128 bits security) - 4-to-1 compression",
        |bench| {
            let v = [Felt::one(); anemoi_4_3::STATE_WIDTH];

            bench.iter(|| anemoi_4_3::AnemoiHash::compress_k(black_box(&v), 4))
        },
    );

    c.bench_function(
        "anemoi-jive/bn_254/6-5 (128 bits security) - 2-to-1 compression",
        |bench| {
            let v = [Felt::one(); anemoi_6_5::STATE_WIDTH];

            bench.iter(|| anemoi_6_5::AnemoiHash::compress(black_box(&v)))
        },
    );

    c.bench_function(
        "anemoi-jive/bn_254/6-5 (128 bits security) - 6-to-1 compression",
        |bench| {
            let v = [Felt::one(); anemoi_6_5::STATE_WIDTH];

            bench.iter(|| anemoi_6_5::AnemoiHash::compress_k(black_box(&v), 6))
        },
    );

    c.bench_function(
        "anemoi-jive/bn_254/8-7 (128 bits security) - 2-to-1 compression",
        |bench| {
            let v = [Felt::one(); anemoi_8_7::STATE_WIDTH];

            bench.iter(|| anemoi_8_7::AnemoiHash::compress(black_box(&v)))
        },
    );

    c.bench_function(
        "anemoi-jive/bn_254/8-7 (128 bits security) - 8-to-1 compression",
        |bench| {
            let v = [Felt::one(); anemoi_8_7::STATE_WIDTH];

            bench.iter(|| anemoi_8_7::AnemoiHash::compress_k(black_box(&v), 8))
        },
    );

    c.bench_function(
        "anemoi-jive/bn_254/10-9 (128 bits security) - 2-to-1 compression",
        |bench| {
            let v = [Felt::one(); anemoi_10_9::STATE_WIDTH];

            bench.iter(|| anemoi_10_9::AnemoiHash::compress(black_box(&v)))
        },
    );

    c.bench_function(
        "anemoi-jive/bn_254/10-9 (128 bits security) - 10-to-1 compression",
        |bench| {
            let v = [Felt::one(); anemoi_10_9::STATE_WIDTH];

            bench.iter(|| anemoi_10_9::AnemoiHash::compress_k(black_box(&v), 10))
        },
    );

    c.bench_function(
        "anemoi-jive/bn_254/12-11 (128 bits security) - 2-to-1 compression",
        |bench| {
            let v = [Felt::one(); anemoi_12_11::STATE_WIDTH];

            bench.iter(|| anemoi_12_11::AnemoiHash::compress(black_box(&v)))
        },
    );

    c.bench_function(
        "anemoi-jive/bn_254/12-11 (128 bits security) - 12-to-1 compression",
        |bench| {
            let v = [Felt::one(); anemoi_12_11::STATE_WIDTH];

            bench.iter(|| anemoi_12_11::AnemoiHash::compress_k(black_box(&v), 12))
        },
    );

    c.bench_function(
        "anemoi-sponge/bn_254/2-1 (128 bits security) - hash 10KB",
        |bench| {
            let mut data = vec![0u8; 10 * 1024];
            let mut rng = OsRng;
            rng.fill_bytes(&mut data);

            bench.iter(|| anemoi_2_1::AnemoiHash::hash(black_box(&data)))
        },
    );

    c.bench_function(
        "anemoi-sponge/bn_254/4-3 (128 bits security) - hash 10KB",
        |bench| {
            let mut data = vec![0u8; 10 * 1024];
            let mut rng = OsRng;
            rng.fill_bytes(&mut data);

            bench.iter(|| anemoi_4_3::AnemoiHash::hash(black_box(&data)))
        },
    );

    c.bench_function(
        "anemoi-sponge/bn_254/6-5 (128 bits security) - hash 10KB",
        |bench| {
            let mut data = vec![0u8; 10 * 1024];
            let mut rng = OsRng;
            rng.fill_bytes(&mut data);

            bench.iter(|| anemoi_6_5::AnemoiHash::hash(black_box(&data)))
        },
    );

    c.bench_function(
        "anemoi-sponge/bn_254/8-7 (128 bits security) - hash 10KB",
        |bench| {
            let mut data = vec![0u8; 10 * 1024];
            let mut rng = OsRng;
            rng.fill_bytes(&mut data);

            bench.iter(|| anemoi_8_7::AnemoiHash::hash(black_box(&data)))
        },
    );

    c.bench_function(
        "anemoi-sponge/bn_254/10_9 (128 bits security) - hash 10KB",
        |bench| {
            let mut data = vec![0u8; 10 * 1024];
            let mut rng = OsRng;
            rng.fill_bytes(&mut data);

            bench.iter(|| anemoi_10_9::AnemoiHash::hash(black_box(&data)))
        },
    );

    c.bench_function(
        "anemoi-sponge/bn_254/12-11 (128 bits security) - hash 10KB",
        |bench| {
            let mut data = vec![0u8; 10 * 1024];
            let mut rng = OsRng;
            rng.fill_bytes(&mut data);

            bench.iter(|| anemoi_12_11::AnemoiHash::hash(black_box(&data)))
        },
    );
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = criterion_benchmark);
criterion_main!(benches);
