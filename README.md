# Anemoi-Rust

This crate provides a Rust implementation of several instantiations of the [Anemoi hash function by Bouvier et al.](https://eprint.iacr.org/2022/840.pdf) over base fields of commonly used curves, backed by the arkworks ecosystem.

It features different instantiations per underlying field, and targets 128 bits security. Each instantiation has an even state size `N`, with a rate size of `N-1`, granted that the present instantiations all work on fields
large enough to ensure 128 bits security with a capacity of a single field element.

**WARNING:** This is an ongoing, prototype implementation subject to changes. In particular, it has not been audited and may contain bugs and security flaws. This implementation is NOT intended for production use.

## Note on no-std usage

* This implementation can be used in `no-std` environments by relying on the `alloc` crate instead. The use of the Rust standard library is activated by default through the `std` feature.

## Fields and instantiations

The currently supported fields are:

* BLS12-377 basefield
* BLS12-381 basefield
* BN-254 basefield
* ED_ON_BLS12-377 basefield (= BLS12-377 scalar field)
* Jubjub basefield (= BLS12-381 scalar field)
* Pallas basefield (= Vesta scalar field)
* Vesta basefield (= Pallas scalar field)

For each of those fields, two instantiations of the Anemoi sponge construction are available:

* 1 column (2 cells) and rate 1
* 2 columns (4 cells) and rate 3

*NOTE*: Thanks to the particular design of the Jive compression mode for Anemoi in Merkle trees configuration, one can put digests both in the capacity and rate registers, where other algebraic hash functions like Rescue-Prime or Poseidon would require a larger number of cells to use their sponge mode as a 2-to-1 compression function to leave the capacity section untouched. In addition, there is almost no overhead of using the Jive compression method with a higher compression factor, reducing the cost of hashing by increasing the Merkle tree arity.

*NOTE*: This implementation here is mostly for illustrative purposes. For a more aggressively optimized version of the Anemoi hash function (and comparison with other hash functions), over the 64 bits "Goldilocks" field
p = 2<sup>64</sup> - 2<sup>32</sup> + 1, one can have a look at this repository: [Toposware/hash](https://github.com/toposware/hash/tree/anemoi). For a comparison of different algebraic
hash functions including Anemoi over the BLS12-381 scalar field, one can have a look at this repository: [dannywillems/ocaml-bls12-381-hash](https://github.com/dannywillems/ocaml-bls12-381-hash).

All instantiations including their test vectors have been generated from the official python reference implementation of Anemoi: [anemoi-hash/anemoi-hash](https://github.com/anemoi-hash/anemoi-hash).

## Build

To build the library with all available instantiations, simply run:

```shell
cargo build --release
```

## Testing

To test all the different instantiations against deterministic test vectors generated from the official SAGEMATH implementation, simply run:

```shell
cargo test --all
```

## Features

By default, all instantiations are available, as well as the Rust standard library. To compile for a no-std environment like WASM, one can turn off the `std` feature
by adding `--no-default-features` when compiling. This will require to manually specify which instantiation we want to access, with which security level. For instance,
to use instances of Anemoi over the BLS12-381 base field without `std` with 128 bits security level, one could compile with:

```shell
cargo build --release --no-default-features --features bls12_381
```

## Performances

In addition to be representable with a short set of constraints in a circuit, making it perfectly suitable for zero-knowledge proof applications, Anemoi native performances compete well with other algebraic hash functions. Below are running times for a security level of 128 bits obtained on an Intel i7-9750H CPU @ 2.60GHz with:

```shell
RUSTFLAGS="-C target-cpu=native" cargo bench --bench bls12_377 --bench vesta
```

### 2-to-1 compression

| Field \ Instantiation | Anemoi-2-1 | Anemoi-4-3 |
| ----------- | ----------- | ----------- |
| BLS12-377 | 429.61 µs | 485.99 µs |
| Vesta | 129.48 µs | 176.58 µs |

### 10KB data hashing

| Field \ Instantiation | Anemoi-2-1 | Anemoi-4-3 |
| ----------- | ----------- | ----------- |
| BLS12-377 | 85.369 ms | 35.937 ms |
| Vesta | 44.448 ms | 20.307 ms |

As expected, the larger the underlying prime field on which we operate, the slower the hash operations get. Seen from the other angle,
FRI-based protocols which do not require an algebraic group can benefit from much more efficient instantiations of Anemoi over smaller fields.

As a comparison, the implementation of Anemoi-8-4 at [Toposware/hash](https://github.com/toposware/hash/tree/anemoi) over the 64 bits "Goldilocks"
field can hash 10KB of data in `1.8249 ms`, i.e. about `24x` and `47x` faster than instantiations with similar internal state byte size,
over Vesta and BLS12-377 base fields, respectively. Perhaps more interestingly, it achieves 2-to-1 compression in `3.9317 µs`, i.e. about
`33x` and `109x` faster than Vesta and BLS12-377 instantiations respectively.

## License

This repository is licensed under:

* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
