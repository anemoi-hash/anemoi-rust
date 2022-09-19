use super::BigInteger256;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi
// [1, 1, 2, 3]
// [3, 1, 1, 2]
// [2, 3, 1, 1]
// [1, 2, 3, 1]
#[allow(unused)]
pub(crate) const MDS: [Felt; NUM_COLUMNS * NUM_COLUMNS] = [
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
        0x00000003fffffffc,
        0xb1096ff400069004,
        0x33189fdfd9789fea,
        0x304962b3598a0adf,
    ])),
    Felt::new(BigInteger256([
        0x00000005fffffffa,
        0x098e27ee0009d806,
        0xcca4efcfc634efe0,
        0x486e140d064f104e,
    ])),
    Felt::new(BigInteger256([
        0x00000005fffffffa,
        0x098e27ee0009d806,
        0xcca4efcfc634efe0,
        0x486e140d064f104e,
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
        0x00000005fffffffa,
        0x098e27ee0009d806,
        0xcca4efcfc634efe0,
        0x486e140d064f104e,
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
        0x00000005fffffffa,
        0x098e27ee0009d806,
        0xcca4efcfc634efe0,
        0x486e140d064f104e,
    ])),
    Felt::new(BigInteger256([
        0x00000001fffffffe,
        0x5884b7fa00034802,
        0x998c4fefecbc4ff5,
        0x1824b159acc5056f,
    ])),
];
