use super::BigInteger256;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi.
/// [1, 2]
/// [2, 5]
#[allow(unused)]
pub(crate) const MDS: [Felt; (NUM_COLUMNS + 1) * (NUM_COLUMNS + 1)] = [
    Felt::new(BigInteger256([
        0x00000001fffffffe,
        0x5884b7fa00034802,
        0x998c4fefecbc4ff5,
        0x1824b159acc5056f,
    ])),
    Felt::new(BigInteger256([
        0x00000003fffffffc,
        0xb1096ff400069004,
        0x33189fdfd9789fea,
        0x304962b3598a0adf,
    ])),
    Felt::new(BigInteger256([
        0x00000003fffffffc,
        0xb1096ff400069004,
        0x33189fdfd9789fea,
        0x304962b3598a0adf,
    ])),
    Felt::new(BigInteger256([
        0x0000000afffffff5,
        0x66d9f3df00120c0b,
        0xcc83b7a7960bb7c5,
        0x04c9cf6d363b9de5,
    ])),
];
