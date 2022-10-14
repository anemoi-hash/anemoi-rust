//! MDS matrix implementation for Anemoi

use super::BigInteger256;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi.
/// [6 1 6]
/// [1 1 5]
/// [5 1 1]
#[allow(unused)]
pub(crate) const MDS: [Felt; NUM_COLUMNS * NUM_COLUMNS] = [
    Felt::new(BigInteger256([
        0x3cf09ab4ffffffe9,
        0xeba8415b2a159e85,
        0xfffffffffffffffc,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x34786d38fffffffd,
        0x992c350be41914ad,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x3cf09ab4ffffffe9,
        0xeba8415b2a159e85,
        0xfffffffffffffffc,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x34786d38fffffffd,
        0x992c350be41914ad,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x34786d38fffffffd,
        0x992c350be41914ad,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0xa1a55e68ffffffed,
        0x74c2a54b4f4982f3,
        0xfffffffffffffffd,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0xa1a55e68ffffffed,
        0x74c2a54b4f4982f3,
        0xfffffffffffffffd,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x34786d38fffffffd,
        0x992c350be41914ad,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x34786d38fffffffd,
        0x992c350be41914ad,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ])),
];
