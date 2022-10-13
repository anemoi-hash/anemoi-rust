use super::BigInteger256;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi
// [ 1  8  7  7]
/// [49 56  8 15]
/// [49 49  1  8]
/// [ 8 15  7  8]
#[allow(unused)]
pub(crate) const MDS: [Felt; NUM_COLUMNS * NUM_COLUMNS] = [
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
        0x0000006bffffff94,
        0xabfecebc00b1306c,
        0x6398dc9befb8ddc0,
        0x17bd68ea718f258a,
    ])),
    Felt::new(BigInteger256([
        0x0000007affffff85,
        0xc3e2328f00c9cc7b,
        0x633534235f3d3570,
        0x4cd09b0b0154ce4f,
    ])),
    Felt::new(BigInteger256([
        0x00000010ffffffef,
        0x70681bcd001be411,
        0x9928a7775c40a7a5,
        0x4d37e37a3c8aae34,
    ])),
    Felt::new(BigInteger256([
        0x00000020ffffffdf,
        0x348ddb9d00362421,
        0x658b26f6c2232750,
        0x0e5d6e47a2b2d9b1,
    ])),
    Felt::new(BigInteger256([
        0x0000006bffffff94,
        0xabfecebc00b1306c,
        0x6398dc9befb8ddc0,
        0x17bd68ea718f258a,
    ])),
    Felt::new(BigInteger256([
        0x0000006bffffff94,
        0xabfecebc00b1306c,
        0x6398dc9befb8ddc0,
        0x17bd68ea718f258a,
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
        0x00000010ffffffef,
        0x70681bcd001be411,
        0x9928a7775c40a7a5,
        0x4d37e37a3c8aae34,
    ])),
    Felt::new(BigInteger256([
        0x00000020ffffffdf,
        0x348ddb9d00362421,
        0x658b26f6c2232750,
        0x0e5d6e47a2b2d9b1,
    ])),
    Felt::new(BigInteger256([
        0x0000000efffffff1,
        0x17e363d300189c0f,
        0xff9c57876f8457b0,
        0x351332208fc5a8c4,
    ])),
    Felt::new(BigInteger256([
        0x00000010ffffffef,
        0x70681bcd001be411,
        0x9928a7775c40a7a5,
        0x4d37e37a3c8aae34,
    ])),
];
