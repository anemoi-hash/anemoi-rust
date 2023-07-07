//! Additive round constants implementation for Anemoi

use super::BigInteger256;
use super::Felt;
use super::NUM_COLUMNS;
use super::NUM_HASH_ROUNDS;

/// Additive round constants C for Anemoi.
pub(crate) const C: [Felt; NUM_COLUMNS * NUM_HASH_ROUNDS] = [
    Felt::new(BigInteger256([
        0x0b0ce7e8ffffff6d,
        0x51762746a8ccf527,
        0xffffffffffffffec,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x6ddbc2775ae1e64e,
        0x493532387365743b,
        0x7465ba349a3d7dff,
        0x369d2381db9addac,
    ])),
    Felt::new(BigInteger256([
        0xc06782695ec9eb24,
        0x890d38c8c464d4d5,
        0x1a4caa8fbcd5d19d,
        0x35723d110821bec9,
    ])),
    Felt::new(BigInteger256([
        0x165b79212d34a3ec,
        0xe10a3f22e4d42e69,
        0xa0630790f9a6117a,
        0x12ce77b5ad7bbbd3,
    ])),
    Felt::new(BigInteger256([
        0x78817653ff6be0d9,
        0x81a69f4d7f450cd7,
        0xd744923c361f9b6a,
        0x0afbbe8518033770,
    ])),
    Felt::new(BigInteger256([
        0xa2c54cb003b36008,
        0xf03315b51a65867b,
        0xa6fa500abcc0d495,
        0x083709719e297528,
    ])),
    Felt::new(BigInteger256([
        0x5fcd2e48568e5ee3,
        0xc8b7bcea8555fcb7,
        0xa4d4f6d10e4528a1,
        0x2b59f32a502b98e5,
    ])),
    Felt::new(BigInteger256([
        0xbf69aa170d574683,
        0x91e711681a5eb680,
        0x3c339c0bfbed0d17,
        0x3193aa5fa71a77f1,
    ])),
    Felt::new(BigInteger256([
        0x2c2ca7a82f04be0d,
        0x7ea536487a11888f,
        0xdee482ac9661839a,
        0x3468093968a345e1,
    ])),
    Felt::new(BigInteger256([
        0xd2fe7dce74443346,
        0xb30efbc93610eabf,
        0xf3155ded0566d50f,
        0x088a5994bca78d32,
    ])),
    Felt::new(BigInteger256([
        0x871637143fb75990,
        0x25b881420f49c51d,
        0xf4415b05ce4edae9,
        0x2a07d022621ddc85,
    ])),
    Felt::new(BigInteger256([
        0xc0e464fb690630dc,
        0xed16fe1fc419524f,
        0xad6614601ef6535c,
        0x35de172d4f116cdf,
    ])),
    Felt::new(BigInteger256([
        0xc57dd1a29ee6d139,
        0xe53d649c324f4d80,
        0x9accce2be6a38f3d,
        0x2f74d48d03e8cafa,
    ])),
    Felt::new(BigInteger256([
        0x29202687d61ada16,
        0x0b9779403bccf723,
        0x2509f4ec4dd24e43,
        0x1aa949d8e3a9149f,
    ])),
    Felt::new(BigInteger256([
        0xbbaf94bfa7d60018,
        0xe3a245502d9cc2e1,
        0x6caedc363b2c54f2,
        0x28d40806dcb2e31c,
    ])),
    Felt::new(BigInteger256([
        0x864a284d5e2259a1,
        0xf9c42416cf2acd71,
        0x1586def4a90ccd5f,
        0x07619650c37c8d1b,
    ])),
    Felt::new(BigInteger256([
        0x5de7b12757bc3840,
        0x2f9a37d2ba0bdbc0,
        0xfefe6f5ded23785f,
        0x356ca8ded5aca8db,
    ])),
    Felt::new(BigInteger256([
        0x3ba0ab7b46a71ce3,
        0x71c57787a14ddc6e,
        0xba527b4c0afaf6a0,
        0x265a122f91c4cfdc,
    ])),
    Felt::new(BigInteger256([
        0xa9bbf8f179ce7b75,
        0x66364a9c18f0f5c0,
        0xca28f26ed2a5879f,
        0x1f35a427ea54c3dc,
    ])),
    Felt::new(BigInteger256([
        0x5258790cccc1edc7,
        0x0d318a1213a25022,
        0xb8549cd96df4a31c,
        0x3beccdd711ae91e7,
    ])),
    Felt::new(BigInteger256([
        0xbec9bc4e0c2aab74,
        0xbb7e3d50de386c01,
        0x63b2e85f530b50bc,
        0x0a9a04ef51428741,
    ])),
    Felt::new(BigInteger256([
        0x8f123ae1c5549a31,
        0x0b0a29d0ebd55b3b,
        0xd5d11c742ac52fb7,
        0x22106b94f916756b,
    ])),
    Felt::new(BigInteger256([
        0xee4c95b6874f8b47,
        0x405d57fe5ace482b,
        0xf52b6381e64c9f0c,
        0x1e507aa3934360ae,
    ])),
    Felt::new(BigInteger256([
        0xddfce35ef1930fb0,
        0x86d4885f84401a4f,
        0xcd157d73831ae8b7,
        0x2a51345b7ad83aa6,
    ])),
    Felt::new(BigInteger256([
        0x4a460bbab9a500ba,
        0x13a2234c1e8f0b56,
        0x5502cdc2d0b4c33a,
        0x381de7018d672a67,
    ])),
    Felt::new(BigInteger256([
        0xde3350fb3e7bd38d,
        0x91e2bfd299da1a3d,
        0xaf2dbe09fa59b2d3,
        0x1356b7042e029abf,
    ])),
    Felt::new(BigInteger256([
        0xf547489dc4b356a8,
        0x6847f2699624ecf4,
        0x4e4301673ee23b80,
        0x29af5acf6f779772,
    ])),
    Felt::new(BigInteger256([
        0xb59bf7586a217691,
        0xb6b5e8a5f3502454,
        0x76c4cd32c6ab3a38,
        0x0c035cf6501f4ab6,
    ])),
];

/// Additive round constants D for Anemoi.
pub(crate) const D: [Felt; NUM_COLUMNS * NUM_HASH_ROUNDS] = [
    Felt::new(BigInteger256([
        0x7c5e333a99999905,
        0xe76b98e699eb6694,
        0xccccccccccccccb8,
        0x0ccccccccccccccc,
    ])),
    Felt::new(BigInteger256([
        0xedd3988924d2d2b4,
        0x179cf9e9cc56ac29,
        0x66ae0597b58a9ce5,
        0x25318a7374ca34ba,
    ])),
    Felt::new(BigInteger256([
        0xb538a036662d31cf,
        0x57c1e2e37215e9f1,
        0xd5b55175ecad2084,
        0x3fcb8bcd293285dc,
    ])),
    Felt::new(BigInteger256([
        0x19d321ae64ef3d65,
        0xe8313f4efa580a06,
        0x81472d0d77fdb27a,
        0x3eef60969aef0d28,
    ])),
    Felt::new(BigInteger256([
        0xbc72050c22d9a9e8,
        0x30914f1fcdfa2528,
        0x2a125fb3f317ea02,
        0x096c722acbf6dff1,
    ])),
    Felt::new(BigInteger256([
        0xf55c662857787be5,
        0xd7901b98d0ed654d,
        0x1f439c18c8397546,
        0x286f573c1e7fa7ea,
    ])),
    Felt::new(BigInteger256([
        0x0b1e5a280780e514,
        0x1c94d4abe2160df4,
        0xc0756ba23b24a120,
        0x25826c054c39a181,
    ])),
    Felt::new(BigInteger256([
        0xe0342fc9eea11f81,
        0xfbefe63ed5a49522,
        0x7d4f8f73774cd7ae,
        0x0d83bd5f6f8b0ace,
    ])),
    Felt::new(BigInteger256([
        0x3f8b840d9eb3db04,
        0x087ca030fa92a8ea,
        0xf882c9b9af688615,
        0x37aa6a3ad71e7af3,
    ])),
    Felt::new(BigInteger256([
        0xf503e4f4144aa30b,
        0x7558bbc31e64d19b,
        0x322f23906cee29a3,
        0x2d9454baf7854c86,
    ])),
    Felt::new(BigInteger256([
        0xda46a62b8bf193f7,
        0x40c8dc4e83587830,
        0x75410cb54cdeb781,
        0x0db0365e854b71c3,
    ])),
    Felt::new(BigInteger256([
        0x22bb5ed2e597be11,
        0x4099af3d9ffacbe4,
        0x53e144a5ec06820e,
        0x3b4e178e3ea18c5e,
    ])),
    Felt::new(BigInteger256([
        0xbe7754780505a69c,
        0x275a45ac4d2c4d8a,
        0xda0391063147fa17,
        0x04c275ae4e74f53c,
    ])),
    Felt::new(BigInteger256([
        0x30c0341d6c910247,
        0x8626b061be7cbdae,
        0x89bc365ce6f70b35,
        0x11be851efa97c922,
    ])),
    Felt::new(BigInteger256([
        0xc78b31f8b14956fa,
        0x181eafd6d8230436,
        0x0e7b8e6fa055b802,
        0x3f18de1c0ffb6edd,
    ])),
    Felt::new(BigInteger256([
        0xa0cc504697ed0351,
        0x66b2e4aee183d547,
        0xdccf0fc45cb68288,
        0x3f6e068ac327a31c,
    ])),
    Felt::new(BigInteger256([
        0xbd841ce928afa16d,
        0xf7cd0804568a48a8,
        0x0e28adfb86eb08b3,
        0x327949c8522dd1f8,
    ])),
    Felt::new(BigInteger256([
        0x10b6711047f1d8dd,
        0x502404ce9c5216bc,
        0xeef8387ff342d90e,
        0x052e4d3ddaa88339,
    ])),
    Felt::new(BigInteger256([
        0x36e3791e5d34a163,
        0x3d733720169c243a,
        0x3c7ec3b1bbf0695e,
        0x22cfa289a878e567,
    ])),
    Felt::new(BigInteger256([
        0x54f9530ce07f6682,
        0xfa9a33ab6fd34c01,
        0x5025ecb2a5bfd6f3,
        0x214e665d9c353db3,
    ])),
    Felt::new(BigInteger256([
        0x246cdec06c4f8847,
        0xe5d4931a509266e2,
        0xf3c725171c5bded4,
        0x2200026ebde51bf0,
    ])),
    Felt::new(BigInteger256([
        0x6a2eb72755d0c9d1,
        0x4b8c3cafbcb52381,
        0x8b60d7c242960fe8,
        0x1b3e0339321b945c,
    ])),
    Felt::new(BigInteger256([
        0xc1c726e7dfaf9646,
        0xef5696372d613703,
        0x663f9eb2a6c21ce7,
        0x0f2c86af2a25edb5,
    ])),
    Felt::new(BigInteger256([
        0xc01dff507a4a6d7d,
        0x6e401ca9bea5cfa8,
        0x63a5373a9210b8ac,
        0x3cf4da8bde1d51ee,
    ])),
    Felt::new(BigInteger256([
        0x381c59056eddee2e,
        0x3951d7261cc66c09,
        0x26605f1dc5e06373,
        0x2076f42c1b86dab8,
    ])),
    Felt::new(BigInteger256([
        0xdab02906240c13cf,
        0xf004c9bdffe44171,
        0xa606cdfb3e05a525,
        0x1d775e538884d551,
    ])),
    Felt::new(BigInteger256([
        0x105009cd401507ad,
        0x607201d267ecc82c,
        0x161a7d62ad4161c0,
        0x31d8d28a17c47837,
    ])),
    Felt::new(BigInteger256([
        0xdf4b434815da7a64,
        0xe7524e202ceac60c,
        0x6417c7c4838ab291,
        0x35f46ed5c4ceb5bc,
    ])),
];
