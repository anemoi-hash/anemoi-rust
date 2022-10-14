//! MDS matrix implementation for Anemoi

use super::BigInteger256;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi.
/// [1   7]
/// [7  50]
#[allow(unused)]
pub(crate) const MDS: [Felt; (NUM_COLUMNS + 1) * (NUM_COLUMNS + 1)] = [
    Felt::new(BigInteger256([
        0x00000001fffffffe,
        0x5884b7fa00034802,
        0x998c4fefecbc4ff5,
        0x1824b159acc5056f,
    ])),
    Felt::new(BigInteger256([
        0x0000000efffffff1,
        0x17e363d300189c0f,
        0xff9c57876f8457b0,
        0x351332208fc5a8c4,
    ])),
    Felt::new(BigInteger256([
        0x0000000efffffff1,
        0x17e363d300189c0f,
        0xff9c57876f8457b0,
        0x351332208fc5a8c4,
    ])),
    Felt::new(BigInteger256([
        0x0000006dffffff92,
        0x048386b600b4786e,
        0xfd252c8bdc752db6,
        0x2fe21a441e542af9,
    ])),
];
