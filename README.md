# Anemoi-hash

This crate provides a Rust implementation of several instantiations of the [Anemoi hash function by Bouvier et al.](https://eprint.iacr.org/2022/840.pdf) over base fields of commonly used curves, backed by the arkworks ecosystem.

It features three different instantiations per underlying field, and targets both 128 or 256 bits security at the user convenience.

* This implementation can be used in `no-std` environments by relying on the `alloc` crate instead. The use of the Rust standard library is activated by default through the `std` feature.

**WARNING:** This is an ongoing, prototype implementation subject to changes. In particular, it has not been audited and may contain bugs and security flaws. This implementation is NOT intended for production use.

The currently supported fields are:

* BLS12-377 basefield
* BLS12-381 basefield
* BN-254 basefield
* Jubjub basefield (= BLS12-381 scalar field)
* Pallas basefield (= Vesta scalar field)

For each of those fields, three instantiations of the Anemoi sponge construction are available:

* 1 column (2 cells) and rate 1
* 4 columns (8 cells) and rate 7
* 6 columns (12 cells) and rate 11

*NOTE*: Thanks to the particular design of the Jive compression mode for Anemoi in Merkle trees configuration, one can put digests both in the capacity and rate registers, where other algebraic hash functions like Rescue-Prime or Poseidon would require a larger number of cells to use their sponge mode as a 2-to-1 compression function to live the capacity section untouched. In addition, there is almost no overhead of using the Jive compression method with a
higher compression factor, reducing the cost of hashing by increasing the Merkle tree arity.

*NOTE*: This implementation here is mostly for illustrative purposes. For a more aggressively optimized version of the Anemoi hash function (and comparison with other hash functions), over the 64 bits "Goldilocks" field
p = 2<sup>64</sup> - 2<sup>32</sup> + 1, one can have a look at this repository: [Toposware/hash](https://github.com/toposware/hash/tree/anemoi). For a comparison of different algebraic
hash functions including Anemoi over the BLS12-381 scalar field, one can have a look at this repository: [dannywillems/ocaml-bls12-381-hash](https://github.com/dannywillems/ocaml-bls12-381-hash).

All instantiations including their test vectors have been generated from this python reference implementation of Anemoi: [Nashtare/anemoi-hash](https://github.com/Nashtare/anemoi-hash).

## Features

By default, all instantiations are available, as well as the Rust standard library. To compile for a no-std environment like WASM, one can turn off the `std` feature
by adding `--no-default-features` when compiling. This will require to manually specify which instantiation we want to access, with which security level. For instance,
to use instances of Anemoi over the BLS12-381 base field without `std`, with 128 bits security level, one could compile with
`cargo build --no-default-features --features bls381,128_bits`.

## Performances

In addition to be representable with a short set of constraints in a circuit, making it perfectly suitable for zero-knowledge proof applications, Anemoi native performances compete well with other algebraic hash functions. Below are running times for a security level of 128 bits obtained on an Intel i7-9750H CPU @ 2.60GHz with `RUSTFLAGS="-C target-cpu=native" cargo bench`:

| Field \ Hash 10KB | Anemoi-2-1 | Anemoi-8-7 | Anemoi-12-11 |
| ----------- | ----------- | -------------- | ---------- |
| BLS12-377 | 85.369 ms | 28.697 ms | 25.178 ms |
| BLS12-381 | 86.979 ms | 27.263 ms | 29.249 ms |
| BN-254 | 46.522 ms | 14.786 ms | 14.387 ms |
| Jubjub | 49.951 ms | 15.640 ms | 15.807 ms |

| Field \ Compression | Anemoi-2-1 | Anemoi-8-7 | Anemoi-12-11 |
| ----------- | ----------- | -------------- | ------------ |
| BLS12-377 | 425.12 µs | 831.42 µs | 1.4192 µs |
| BLS12-381 | 397.67 µs | 842.77 µs | 1.2744 ms |
| BN-254 | 148.34 µs | 329.67 µs | 468.49 µs |
| Jubjub | 152.35 µs | 324.26 µs | 492.20 µs |

As expected, the larger the underlying prime field on which we operate, the slower the hash operations get. Seen from the other angle, FRI-based protocols which do not require an algebraic group can benefit from much more efficient instantiations of Anemoi over smaller fields. As a comparison, the implementation of Anemoi-8-7 
at [Toposware/hash](https://github.com/toposware/hash/tree/anemoi) over the 64 bits "Goldilocks" field can hash 10KB of data in 951.71 µs, i.e. about 16x and 29x faster than the same instantiations over Jubjub and BLS12-381 base fields, respectively.

## License

This repository is licensed under:

* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
