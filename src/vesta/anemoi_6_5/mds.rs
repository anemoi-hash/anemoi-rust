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
        0x65a0e008ffffffe9,
        0xeba8415b23a4d418,
        0xfffffffffffffffc,
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
        0x5b2b3e9cfffffffd,
        0x992c350be3420567,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x5b2b3e9cfffffffd,
        0x992c350be3420567,
        0xffffffffffffffff,
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
        0x5b2b3e9cfffffffd,
        0x992c350be3420567,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x5b2b3e9cfffffffd,
        0x992c350be3420567,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ])),
];
