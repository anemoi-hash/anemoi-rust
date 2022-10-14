//! MDS matrix implementation for Anemoi

use super::BigInteger256;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi.
/// [8 1 8]
/// [1 1 7]
/// [7 1 1]
#[allow(unused)]
pub(crate) const MDS: [Felt; NUM_COLUMNS * NUM_COLUMNS] = [
    Felt::new(BigInteger256([
        0x00000010ffffffef,
        0x70681bcd001be411,
        0x9928a7775c40a7a5,
        0x4d37e37a3c8aae34,
    ])),
    Felt::new(BigInteger256([
        0x00000001fffffffe,
        0x5884b7fa00034802,
        0x998c4fefecbc4ff5,
        0x1824b159acc5056f,
    ])),
    Felt::new(BigInteger256([
        0x00000010ffffffef,
        0x70681bcd001be411,
        0x9928a7775c40a7a5,
        0x4d37e37a3c8aae34,
    ])),
    Felt::new(BigInteger256([
        0x00000001fffffffe,
        0x5884b7fa00034802,
        0x998c4fefecbc4ff5,
        0x1824b159acc5056f,
    ])),
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
        0x00000001fffffffe,
        0x5884b7fa00034802,
        0x998c4fefecbc4ff5,
        0x1824b159acc5056f,
    ])),
    Felt::new(BigInteger256([
        0x00000001fffffffe,
        0x5884b7fa00034802,
        0x998c4fefecbc4ff5,
        0x1824b159acc5056f,
    ])),
];
