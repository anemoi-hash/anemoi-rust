//! Additive round constants implementation for Anemoi

use super::BigInteger256;
use super::Felt;
use super::{NUM_COLUMNS, NUM_HASH_ROUNDS};

/// Additive round constants C for Anemoi.
pub(crate) const C: [[Felt; NUM_COLUMNS]; NUM_HASH_ROUNDS] = [
    [
        Felt::new(BigInteger256([
            0x8c7ac37d8dacf7b1,
            0x8379fceb85c49fef,
            0xcdfd9f1c6987245d,
            0x07834efb432341d9,
        ])),
        Felt::new(BigInteger256([
            0xbcb4a9832e0452f6,
            0x17f32b51a1825b47,
            0xbc7cfa76f63d4d10,
            0x2080396d88a28da5,
        ])),
        Felt::new(BigInteger256([
            0x58383a2b3e062e78,
            0x139087bf6787bba0,
            0x746f4ef9c0075f89,
            0x0c1e6f9df6e100ec,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0x4a352612c6a8e3f0,
            0x9dc26d51a557485b,
            0xefb81e824a630f77,
            0x1897a39081505f21,
        ])),
        Felt::new(BigInteger256([
            0xaf8f311d307dde18,
            0x2dc337dea19c5d6d,
            0x8bca456f059100d0,
            0x2e056bfa2066b103,
        ])),
        Felt::new(BigInteger256([
            0x5d3ccb501a400bfe,
            0xab9ae8a14e3a9ca5,
            0x9f6525bedf2f1f8c,
            0x170918aab2d1b637,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xf7d8d3b737ea1d29,
            0xdcc2dcdfea9edeef,
            0xa9821f4b970c5434,
            0x0880c672be1b35ac,
        ])),
        Felt::new(BigInteger256([
            0x9b033bfba8347d8e,
            0x4989cfaefada94a3,
            0x5644e18871b8aa8f,
            0x0760d0b4ec5e688d,
        ])),
        Felt::new(BigInteger256([
            0xaf8d70e258694b4b,
            0xc694d35cfcce72d5,
            0x21d62af141d3012a,
            0x0db8ee2033a275c2,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xcfe134c7cab7cb71,
            0xb40cb4c365f49ee2,
            0x8a2d45ec04b07066,
            0x1c144fb74db40686,
        ])),
        Felt::new(BigInteger256([
            0xfd2c413f8aadf0dc,
            0x25a9b563b789d1ee,
            0x8443ae3ef1cd82d6,
            0x1812e17b35225e1f,
        ])),
        Felt::new(BigInteger256([
            0xdd5e3343bc880e9f,
            0xf6ed2f6515c39c41,
            0xb7d0e88824d9c7d8,
            0x20abc3aadb358361,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xa39878d923e054dd,
            0x0bd17b23ef33a980,
            0x10b2798eae5a9df3,
            0x04a15145df894402,
        ])),
        Felt::new(BigInteger256([
            0x46d8fca9d3a398a0,
            0x6f8eda409a49ca67,
            0x0d37d68dad44e002,
            0x223844d31901ddbc,
        ])),
        Felt::new(BigInteger256([
            0x7257ccaf4f0b6412,
            0xf469403345675112,
            0x860c0eee10d3042f,
            0x2a47ee26a14411a9,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0x350c014927a20623,
            0x25d0f48d0f9462b5,
            0x068ec9b27486a617,
            0x014c45b801fe46bd,
        ])),
        Felt::new(BigInteger256([
            0xd659fbb7ab7f255a,
            0x21e0a21f7edb5323,
            0xc5ee450c5743164b,
            0x0953c3786439b0b8,
        ])),
        Felt::new(BigInteger256([
            0xe3b1cf89618a78bc,
            0xbe83467c65286167,
            0xd7271e0ba475182e,
            0x09fa0b6770401baa,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xd6c4ed792a61ed79,
            0xe57202db8eaaee4c,
            0x81a11a2c7ba52d13,
            0x068b2baa0cc4a6b4,
        ])),
        Felt::new(BigInteger256([
            0x13778194c8698de2,
            0xd77218dcf8f01f07,
            0x78c8a354c9a47153,
            0x220a2e8bb81e9d99,
        ])),
        Felt::new(BigInteger256([
            0x90a7c328a4ab0f61,
            0x93255ed99ec1591f,
            0xd42423690ba1a321,
            0x20c4d3e150494f99,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xd41dd442ab91976b,
            0x9f90463e3cf35c61,
            0x576cfa3c91fc74bd,
            0x1b9a510a722108df,
        ])),
        Felt::new(BigInteger256([
            0x6578e5ec5996b67d,
            0x23b9b04b9e02384a,
            0x7fe603f559d48d86,
            0x234b7978afd113e4,
        ])),
        Felt::new(BigInteger256([
            0xe30e490d0ee49729,
            0xd73fd9eab4be31ed,
            0xf0ac81f5f063696a,
            0x17b39d149114cf7f,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xcb6309851cb51205,
            0x9a9622f96760fa86,
            0xb806e9d7a219287c,
            0x1bdc93ec06e2adb9,
        ])),
        Felt::new(BigInteger256([
            0x02c59b21cc7c43b1,
            0xc4ed6d619a817baf,
            0x46832bf55f34cde4,
            0x2afff13fbc0220ae,
        ])),
        Felt::new(BigInteger256([
            0xebc7d154bd739901,
            0xa8d0a64072a09d7a,
            0x9d130291903bc9b4,
            0x0f8f39ae176003b3,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0x2af89d2b0f05d8b0,
            0x7862ad62dc2b173e,
            0xd0b8ab58c35ba407,
            0x257b41417e5b8751,
        ])),
        Felt::new(BigInteger256([
            0x5315bddfd95b8722,
            0x620e69e6f79f7698,
            0xc32bbed12b8ec0bc,
            0x19d298b934cb2d4c,
        ])),
        Felt::new(BigInteger256([
            0xb689d5bfebb3dbe9,
            0x6f470683a1347879,
            0xa0b55767bc93ffeb,
            0x1eb8785bcee3c03b,
        ])),
    ],
];
/// Additive round constants D for Anemoi.
pub(crate) const D: [[Felt; NUM_COLUMNS]; NUM_HASH_ROUNDS] = [
    [
        Felt::new(BigInteger256([
            0x3c4f5e09c25ba5fd,
            0x6422c429a52ce733,
            0x82e7ecd3bcdaae8b,
            0x1c52e66216e116f7,
        ])),
        Felt::new(BigInteger256([
            0x99c52c2526d457e8,
            0xa0e1eda4470f81a4,
            0x4d7a95c913c9cd15,
            0x19b801ad712cecba,
        ])),
        Felt::new(BigInteger256([
            0x5f5c663713603065,
            0x0f7fff14d90c2975,
            0x2269899a0a3e6b83,
            0x2597d27153d70b67,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xa731b3d81fb2d234,
            0xc47aa4d60a3a7601,
            0xcf79f8b21bf685b9,
            0x0c363e1d8d409d90,
        ])),
        Felt::new(BigInteger256([
            0x39c7a6f84da92302,
            0xfcc16a778ca46a2d,
            0x479f6d39a15d6ce9,
            0x060c376041237969,
        ])),
        Felt::new(BigInteger256([
            0x1188ea9513f54de3,
            0xed99d03d0539f0dd,
            0x7836ecd7a7a6179a,
            0x0f517ea447fa2a03,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0x76268a752861db39,
            0xbc9113ed82889a89,
            0x797351d166c2e70d,
            0x2d4274511143a45f,
        ])),
        Felt::new(BigInteger256([
            0x468cdacf5ccd9244,
            0xd19e01d118e92f56,
            0x024961a90ba8333f,
            0x108aaf6c54536137,
        ])),
        Felt::new(BigInteger256([
            0x490a2d09110f5fb5,
            0x2a284ff07e628a73,
            0x328704a986ebbd72,
            0x06c018f82ed179a8,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0x6fe12ea447c6f942,
            0xed1f75d65d45c1c8,
            0xb2499a8e67fb10af,
            0x1611356c5fa0dd07,
        ])),
        Felt::new(BigInteger256([
            0x0688af48a45b729a,
            0x9e83dc1c9d719e7b,
            0x40c39632a0d27154,
            0x26dc467c3d0d5ec1,
        ])),
        Felt::new(BigInteger256([
            0xd4adbe9fda429011,
            0x4b46a08f5f30e5b8,
            0xd8fd2a137f07e9ee,
            0x1f5274cc765a8f3f,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0x68ff25ed9d151663,
            0xcf4380aa1ab729c5,
            0xb32cc3bf19dde0b3,
            0x133e55cc92c1622c,
        ])),
        Felt::new(BigInteger256([
            0x397b91d410f9b0cc,
            0xdb46dadb4bf229c5,
            0x8bc56e58e301189a,
            0x153d7a32e10685dd,
        ])),
        Felt::new(BigInteger256([
            0x52ed7f2c906e7bf2,
            0x3ba08b3f5a952d5b,
            0x69460050f1b8705f,
            0x0d2a6fa6fc82c507,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xcaa96a45b9c6e984,
            0xcfd22e75d09ade93,
            0x3606a9e8d6e761e6,
            0x230f89fba23cc946,
        ])),
        Felt::new(BigInteger256([
            0x99334cca01c55f61,
            0x7427d71cc606ae1b,
            0xd17972dd83dcc7f2,
            0x0f7f38951944bd38,
        ])),
        Felt::new(BigInteger256([
            0x947e3deebbddb277,
            0xec49c5eb0fd9394a,
            0x475ea5747c37fd6c,
            0x0002cca4b8853367,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0x389eb29789a9b320,
            0x41197a396620479d,
            0x2a064422ddacd464,
            0x09f7ac60efcd33d0,
        ])),
        Felt::new(BigInteger256([
            0xa28d2ec8ebd2aa2f,
            0xdb5f8b4f568a5770,
            0xfd411ae5f5e50e7b,
            0x09dee01baff3b4ab,
        ])),
        Felt::new(BigInteger256([
            0x49d119c6a49e28a9,
            0x0a13864ec852d901,
            0x75993a48648ccc3e,
            0x28db2004bc8a1212,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xe9d6c70c63bd434f,
            0x994d7e1115edf420,
            0x39158c15e612d42f,
            0x153b64339d93cbb4,
        ])),
        Felt::new(BigInteger256([
            0xa86dc0cbd5e3b907,
            0xc5bce332fd21af22,
            0x3da1e3697823e2cf,
            0x0154bd7af01060b0,
        ])),
        Felt::new(BigInteger256([
            0x5016cd5667bb96ae,
            0xec43c1d4dfd4f03e,
            0xcb6500b83b5d4aa8,
            0x15fe7baa45bfc7b1,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0x0e3484f1441a9326,
            0x71535b99447ab6f5,
            0xe7a13f6a45061f41,
            0x173f0d95d0f59bb4,
        ])),
        Felt::new(BigInteger256([
            0x72d2fea3b8031b78,
            0x43f0a115fdc01736,
            0x5230cf22cc5aba81,
            0x0aca9bc29ae198a0,
        ])),
        Felt::new(BigInteger256([
            0x85e8de4085846dc3,
            0x9ad48ef7a1d6807a,
            0xc5bd450d2a0c4245,
            0x0f9b7ec46aab270b,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0x07fa10bae2e04fce,
            0x9fa304eb832414d8,
            0x1dfaca20bfcf02b5,
            0x2989541ec0502ea0,
        ])),
        Felt::new(BigInteger256([
            0x5d531985715754e6,
            0x3194bc8424bd534b,
            0xec812b33f23b1542,
            0x0248dc6f8b8c5e91,
        ])),
        Felt::new(BigInteger256([
            0xeadadacf6039a6a8,
            0xb1ce0e239a499ca4,
            0xe7076318afeae065,
            0x277056a59a109ce6,
        ])),
    ],
];
