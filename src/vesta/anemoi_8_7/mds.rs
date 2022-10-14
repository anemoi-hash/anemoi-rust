//! MDS matrix implementation for Anemoi

use super::BigInteger256;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi
/// [ 1  6  5  5]
/// [25 30  6 11]
/// [25 25  1  6]
/// [ 6 11  5  6]
#[allow(unused)]
pub(crate) const MDS: [Felt; NUM_COLUMNS * NUM_COLUMNS] = [
    Felt::new(BigInteger256([
        0x5b2b3e9cfffffffd,
        0x992c350be3420567,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x65a0e008ffffffe9,
        0xeba8415b23a4d418,
        0xfffffffffffffffc,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x96bc8c8cffffffed,
        0x74c2a54b49f7778e,
        0xfffffffffffffffd,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x96bc8c8cffffffed,
        0x74c2a54b49f7778e,
        0xfffffffffffffffd,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0xc093123cffffff9d,
        0xbeb2d6884b82b252,
        0xfffffffffffffff2,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0xcb08b3a8ffffff89,
        0x112ee2d78be58103,
        0xfffffffffffffff0,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x65a0e008ffffffe9,
        0xeba8415b23a4d418,
        0xfffffffffffffffc,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x70168174ffffffd5,
        0x3e244daa6407a2c9,
        0xfffffffffffffffa,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0xc093123cffffff9d,
        0xbeb2d6884b82b252,
        0xfffffffffffffff2,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0xc093123cffffff9d,
        0xbeb2d6884b82b252,
        0xfffffffffffffff2,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x5b2b3e9cfffffffd,
        0x992c350be3420567,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x65a0e008ffffffe9,
        0xeba8415b23a4d418,
        0xfffffffffffffffc,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x65a0e008ffffffe9,
        0xeba8415b23a4d418,
        0xfffffffffffffffc,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x70168174ffffffd5,
        0x3e244daa6407a2c9,
        0xfffffffffffffffa,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x96bc8c8cffffffed,
        0x74c2a54b49f7778e,
        0xfffffffffffffffd,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x65a0e008ffffffe9,
        0xeba8415b23a4d418,
        0xfffffffffffffffc,
        0x3fffffffffffffff,
    ])),
];
