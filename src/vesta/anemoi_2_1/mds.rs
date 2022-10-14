//! MDS matrix implementation for Anemoi

use super::BigInteger256;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi.
/// [1, 5]
/// [5, 26]
#[allow(unused)]
pub(crate) const MDS: [Felt; (NUM_COLUMNS + 1) * (NUM_COLUMNS + 1)] = [
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
        0x8f7765b8ffffff99,
        0x3598729825300edc,
        0xfffffffffffffff2,
        0x3fffffffffffffff,
    ])),
];
