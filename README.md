# Anemoi-hash

This crate provides a Rust implementation of several instantiations of the [Anemoi hash function by Bouvier et al.](https://eprint.iacr.org/2022/840.pdf) over base fields of commonly used curves, backed by the arkworks ecosystem.

It features different instantiations per underlying field, and targets 128 bits security. Each instantiation has an even state size `N`, with a rate size of `N-1`, granted that the present instantiations all work on fields
large enough to ensure 128 bits security with a capacity of a single field element.

* This implementation can be used in `no-std` environments by relying on the `alloc` crate instead. The use of the Rust standard library is activated by default through the `std` feature.

**WARNING:** This is an ongoing, prototype implementation subject to changes. In particular, it has not been audited and may contain bugs and security flaws. This implementation is NOT intended for production use.

The currently supported fields are:

* BLS12-377 basefield
* BLS12-381 basefield
* BN-254 basefield
* ED_ON_BLS12-377 basefield (= BLS12-377 scalar field)
* Jubjub basefield (= BLS12-381 scalar field)
* Pallas basefield (= Vesta scalar field)
* Vesta basefield (= Pallas scalar field)

For each of those fields, three instantiations of the Anemoi sponge construction are available:

* 1 column (2 cells) and rate 1
* 2 columns (4 cells) and rate 3
* 3 columns (6 cells) and rate 5
* 4 columns (8 cells) and rate 7
* 6 columns (12 cells) and rate 11

*NOTE*: Thanks to the particular design of the Jive compression mode for Anemoi in Merkle trees configuration, one can put digests both in the capacity and rate registers, where other algebraic hash functions like Rescue-Prime or Poseidon would require a larger number of cells to use their sponge mode as a 2-to-1 compression function to live the capacity section untouched. In addition, there is almost no overhead of using the Jive compression method with a
higher compression factor, reducing the cost of hashing by increasing the Merkle tree arity.

*NOTE*: This implementation here is mostly for illustrative purposes. For a more aggressively optimized version of the Anemoi hash function (and comparison with other hash functions), over the 64 bits "Goldilocks" field
p = 2<sup>64</sup> - 2<sup>32</sup> + 1, one can have a look at this repository: [Toposware/hash](https://github.com/toposware/hash/tree/anemoi). For a comparison of different algebraic
hash functions including Anemoi over the BLS12-381 scalar field, one can have a look at this repository: [dannywillems/ocaml-bls12-381-hash](https://github.com/dannywillems/ocaml-bls12-381-hash).

All instantiations including their test vectors have been generated from the official python reference implementation of Anemoi: [vesselinux/anemoi-hash](https://github.com/vesselinux/anemoi-hash).

## Features

By default, all instantiations are available, as well as the Rust standard library. To compile for a no-std environment like WASM, one can turn off the `std` feature
by adding `--no-default-features` when compiling. This will require to manually specify which instantiation we want to access, with which security level. For instance,
to use instances of Anemoi over the BLS12-381 base field without `std` with 128 bits security level, one could compile with
`cargo build --no-default-features --features bls12_381`.

## Performances

In addition to be representable with a short set of constraints in a circuit, making it perfectly suitable for zero-knowledge proof applications, Anemoi native performances compete well with other algebraic hash functions. Below are running times for a security level of 128 bits obtained on an Intel i7-9750H CPU @ 2.60GHz with `RUSTFLAGS="-C target-cpu=native" cargo bench`:

| Field \ Compression | Anemoi-2-1 | Anemoi-4-3 | Anemoi-8-7 | Anemoi-12-11 |
| ----------- | ----------- | ----------- | -------------- | ------------ |
| BLS12-377 | 396.41 µs | 493.55 µs | 818.02 µs | 1.2514 µs |
| BLS12-381 | 433.43 µs | 541.73 µs | 810.05 µs | 1.4194 ms |
| BN-254 | 142.40 µs | 179.67 µs | 330.81 µs | 448.30 µs |
| ED on BLS12-377 | 157.84 µs | 191.61 µs | 304.42 µs | 453.30 µs |
| Jubjub | 170.72 µs | 229.40 µs | 335.26 µs | 506.81 µs |
| Pallas | 141.41 µs | 170.97 µs | 291.87 µs | 435.68 µs |
| Vesta | 129.48 µs | 186.36 µs | 285.26 µs | 440.80 µs |

| Field \ Hash 10KB | Anemoi-2-1 | Anemoi-4-3 | Anemoi-8-7 | Anemoi-12-11 |
| ----------- | ----------- | ----------- | -------------- | ---------- |
| BLS12-377 | 85.369 ms | 35.937 ms | 27.206 ms | 28.603 ms |
| BLS12-381 | 86.478 ms | 41.431 ms | 26.143 ms | 24.794 ms |
| BN-254 | 53.219 ms | 22.458 ms | 14.549 ms | 14.588 ms |
| ED on BLS12-377 | 50.076 ms | 28.744 ms | 17.053 ms | 16.137 ms |
| Jubjub | 54.205 ms | 22.902 ms | 16.573 ms | 16.077 ms |
| Pallas | 54.827 ms | 19.818 ms | 15.905 ms | 15.370 ms |
| Vesta | 47.268 ms | 20.109 ms | 14.361 ms | 14.113 ms |

As expected, the larger the underlying prime field on which we operate, the slower the hash operations get. Seen from the other angle, FRI-based protocols which do not require an algebraic group can benefit from much more efficient instantiations of Anemoi over smaller fields. As a comparison, the implementation of Anemoi-8-7
at [Toposware/hash](https://github.com/toposware/hash/tree/anemoi) over the 64 bits "Goldilocks" field can hash 10KB of data in 1.231 ms, i.e. about 44x and 70x faster than the same instantiations over Jubjub and BLS12-381 base fields, respectively.

## License

This repository is licensed under:

* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
