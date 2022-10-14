//! Additive round constants implementation for Anemoi

use super::BigInteger256;
use super::Felt;
use super::{NUM_COLUMNS, NUM_HASH_ROUNDS};

/// Additive round constants C for Anemoi.
pub(crate) const C: [[Felt; NUM_COLUMNS]; NUM_HASH_ROUNDS] = [
    [
        Felt::new(BigInteger256([
            0xabafffffffff9120,
            0x4bb663a9ffff8cc3,
            0xe91979cf61a56a4a,
            0x0478a6f9e9e436dd,
        ])),
        Felt::new(BigInteger256([
            0xa36f1774329ceb88,
            0x28c335a5daf88982,
            0x5c91edea0ad0b3f7,
            0x020ad5f2bd639b37,
        ])),
        Felt::new(BigInteger256([
            0x03a61e2f43065b73,
            0x60de485cd8f04488,
            0xffd33caaa40ccaf5,
            0x0ed83a71171f901b,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xbfdc959fc7b4b4f6,
            0xa1645fd4ff9605ce,
            0xd5691b0520c7782e,
            0x0a14dedd9894572a,
        ])),
        Felt::new(BigInteger256([
            0x3fa34fcfa658b7b6,
            0x926fbf1bfad80b3f,
            0xfd04325b984e4684,
            0x0e9782736f581fef,
        ])),
        Felt::new(BigInteger256([
            0x4e99c9e56474f6d0,
            0x73fecfcd343516b6,
            0x359b9b34d37d43f1,
            0x0ad1b37360dc6fa8,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xa8710aeddc00c31f,
            0x48d555e0b98bbcdc,
            0x90c713f2837ff521,
            0x0a1f01b2d09a9963,
        ])),
        Felt::new(BigInteger256([
            0xe657fdc69ffcdada,
            0x6e84f0fa5ab7542c,
            0x2104c49ea979958a,
            0x05403d5ed50440f7,
        ])),
        Felt::new(BigInteger256([
            0x33e98478dc36fb0b,
            0x4cb464cb80f70699,
            0xf5ede43f92f6a193,
            0x00e198e22427a530,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xa747c810b42443fb,
            0xf6d421a4dfe10249,
            0xc87bd48e64d5d249,
            0x0b6b19608e0494d1,
        ])),
        Felt::new(BigInteger256([
            0x3f1d72712937bbaa,
            0x3a9f2e75f5d7855f,
            0x07951c993e46fa3a,
            0x007308631618c1ed,
        ])),
        Felt::new(BigInteger256([
            0x047d7ae2d820611f,
            0xe56d2726cbcebab4,
            0xdcc1106a538c9676,
            0x057f727bb3c1fd0d,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0x4b84564b4ed4cc40,
            0x614fb3d8ba6fa258,
            0x6d21dc5b9cbc24a9,
            0x0d7804aeee114147,
        ])),
        Felt::new(BigInteger256([
            0xe01e615d06512a64,
            0xa28cb477c1492122,
            0xba7d2e5c4fe4cef3,
            0x05e902094d782d9e,
        ])),
        Felt::new(BigInteger256([
            0x8b1b907b0d2aabe2,
            0x2b85a7527e3a68a2,
            0x2e1297ffc7a7fde9,
            0x0c0c5bfda9df4ff1,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0x094d0862efc8d4b2,
            0x518ff99b3760e6b7,
            0x14b68a84040da5fe,
            0x0b4b897ead371aa6,
        ])),
        Felt::new(BigInteger256([
            0x52aea826b11aadff,
            0x929582b0645b6b26,
            0xa01402c60eae13b4,
            0x100fc88c161df24b,
        ])),
        Felt::new(BigInteger256([
            0x331a9b31635a642f,
            0x4b8d56cd0404e94c,
            0x3b5eac9faadcd55c,
            0x03dc6d1817708eb0,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xb65d03158e8d330a,
            0x062c677ab9fb38fa,
            0x395365290dc341a3,
            0x075e6fec1dbc6a05,
        ])),
        Felt::new(BigInteger256([
            0x94721b95c7cb761d,
            0x84d886e209688d3b,
            0xb7ba836bf79463e2,
            0x0af0a069cd1110b1,
        ])),
        Felt::new(BigInteger256([
            0x00ad41972663719f,
            0xf641ae6f2b56b0e4,
            0x73339fc3638e7d75,
            0x0f40f5dbb062a474,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0x405cc568b090bbf5,
            0x90216c2a4a1335e3,
            0xdabc18b784bdda4e,
            0x0c3faca6b6e1fb47,
        ])),
        Felt::new(BigInteger256([
            0x3079e6f2573ff59e,
            0x855611e8f887ab17,
            0x288fcf9369d2a90d,
            0x12601cffbbd2fbf8,
        ])),
        Felt::new(BigInteger256([
            0x62e081a2bde42899,
            0x9b0ec406f8d0ad7a,
            0x93c9412a72a09702,
            0x075d486e813432f8,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0x1ce64c4fbff3bf46,
            0x9ef195190fc39b99,
            0x4f7f6c077b257685,
            0x0af1b0422db4e68e,
        ])),
        Felt::new(BigInteger256([
            0xc6edee6afd0b777e,
            0x1b9266ccc595d5e8,
            0x0d6bc08b6d0675e9,
            0x0e073b16fed33b0f,
        ])),
        Felt::new(BigInteger256([
            0x62215151561bcf71,
            0x694f0c663aae25d5,
            0x26254f47b73a9e19,
            0x0a84c08ecfabb160,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xd81c4503f4186b9b,
            0xf3e86355d2c3a343,
            0xa8655929d372f300,
            0x0bf60b0e67b9f46c,
        ])),
        Felt::new(BigInteger256([
            0xfbbc1438b710430c,
            0xcd9b2c6ef7ce1746,
            0x80d7948d66a027bd,
            0x0ce0f616bfc893a6,
        ])),
        Felt::new(BigInteger256([
            0x14d1e986203f09b9,
            0x6170c7996b0e2fd9,
            0x4c89118573cc17fb,
            0x03004d945a6414ba,
        ])),
    ],
];
/// Additive round constants D for Anemoi.
pub(crate) const D: [[Felt; NUM_COLUMNS]; NUM_HASH_ROUNDS] = [
    [
        Felt::new(BigInteger256([
            0x631f9745d173ee37,
            0x4a87e61eafff8cc3,
            0xe5ff9352bcdc0bbe,
            0x0b0416f762c0c3f4,
        ])),
        Felt::new(BigInteger256([
            0x912b24914e2783db,
            0x0b810b6bebc95515,
            0x74d39a137bdb3a53,
            0x0ea663ad0ca26cdc,
        ])),
        Felt::new(BigInteger256([
            0x92b609578f1254df,
            0xd751c91e6886de13,
            0x7bd080bbcdda3b49,
            0x0aec23bb486dcb0f,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0x87f729b70dd07c39,
            0x834a0803a0820bb8,
            0x317e1fc7d2e63bf4,
            0x0d71b42dcbdf1706,
        ])),
        Felt::new(BigInteger256([
            0x33f8d9be368aba34,
            0xfe97439d2c94dcbb,
            0x13c07ca604093f30,
            0x05591021ded87f03,
        ])),
        Felt::new(BigInteger256([
            0xee54b1df25285a68,
            0xcd867648b4b7b62b,
            0x10c7ca855432d697,
            0x03b702104c98dd60,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xf4c0bcf4a6c3cc60,
            0xd8d96837dbe528c6,
            0x69b35452c2620234,
            0x0badb273ea671d56,
        ])),
        Felt::new(BigInteger256([
            0x68f425a4b4d61f57,
            0xe27556a2dde18baa,
            0x154c97a4fe2f8785,
            0x0cdf0bdcc5330978,
        ])),
        Felt::new(BigInteger256([
            0x61eb0a622191a0a2,
            0xae04ec6e52e70c10,
            0xaea59c4bfca72d88,
            0x0aa4284e90927c56,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0x900b244fb0904625,
            0xb096a2690fc1e758,
            0x6a6bf60727070908,
            0x041ee3f01f042972,
        ])),
        Felt::new(BigInteger256([
            0x683ec4876fb9f911,
            0x31f8798a56893602,
            0x25951dd67283c5e2,
            0x11e2560e17a74072,
        ])),
        Felt::new(BigInteger256([
            0xcef2ab044f23ff9f,
            0x707c1d36ab46394f,
            0x5e7ca98f408c4c17,
            0x06671bb6975fe4e1,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0x9ac5ecc7cbcdbdce,
            0x1c3798121f107e06,
            0xd370b8d18d209d77,
            0x0ee4a9c71adbe1a8,
        ])),
        Felt::new(BigInteger256([
            0x65ac6db0cd60572e,
            0x4160ec0286bac864,
            0x3c279d78561d2ca9,
            0x0d65c4de50a5128e,
        ])),
        Felt::new(BigInteger256([
            0xb1fd7ada04bb39c5,
            0x5e0f89d8c271dddc,
            0x13789f0386a34597,
            0x03017a628f1b9e2f,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xd43665b6caf64b48,
            0xbbdc285be039533f,
            0x0327decf2d91be11,
            0x1054f5b00a8be98e,
        ])),
        Felt::new(BigInteger256([
            0x49d2fb51d65e5fd0,
            0x87238dc39e04a341,
            0x492c9c98f1ce60ae,
            0x087ded1bafa8606b,
        ])),
        Felt::new(BigInteger256([
            0xdfb5cc67b91f771b,
            0x8725fad95c73ef61,
            0x099b7896ff2f6c51,
            0x1119b7f4c763b0cb,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xcc9541c055d23450,
            0x0b1903f37e920489,
            0xe9f35fa9483e30d9,
            0x123e2e636f1b0391,
        ])),
        Felt::new(BigInteger256([
            0xd6e55017d926b29e,
            0x1406ffad5ed0245c,
            0x2301c373ebab87ff,
            0x0935173f5aa54976,
        ])),
        Felt::new(BigInteger256([
            0xee85d42468400f3a,
            0x72d04934cf8415fe,
            0xa2eac4d16ca03b8c,
            0x0fa92d9fba32ebdd,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xc2b22d9af007839b,
            0xa6095ae2c11500d7,
            0xea02a77c314478d6,
            0x024c9ce75cfe9049,
        ])),
        Felt::new(BigInteger256([
            0xe91bc4fbe0ccf880,
            0x7f2a53f2d05a419e,
            0x5331f0fe2c2d2c7d,
            0x0e7d2afd3851d588,
        ])),
        Felt::new(BigInteger256([
            0xc6e7bdb777f28c95,
            0x8243280b1f6911fb,
            0x82db479b49f5b46c,
            0x059e175a79ef1b2d,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0x23ddc3fffe4a2579,
            0xa7fd28b2360523b3,
            0xae0096e00d9ab955,
            0x087cc0ab19d909c3,
        ])),
        Felt::new(BigInteger256([
            0x0431dbf2857818ed,
            0x088a4db74ca82996,
            0x87487e0a154f9da1,
            0x11a2693cc159a2d2,
        ])),
        Felt::new(BigInteger256([
            0x4aca9ce40f09d1fa,
            0x43a7154b1086477c,
            0x6471f1cc747e5fcb,
            0x1043afa30e6e27c8,
        ])),
    ],
    [
        Felt::new(BigInteger256([
            0xd3c34969c9626da6,
            0x5f45bdcc3741afc8,
            0x31dd3ecec0046cb3,
            0x0ace99c7ee0eaafa,
        ])),
        Felt::new(BigInteger256([
            0x2daf8e75d6708053,
            0x1ce4da36bd1cef5f,
            0x25ab0cd869058658,
            0x11c9a28d1c7f8ec2,
        ])),
        Felt::new(BigInteger256([
            0xf22ac1ce7020a81a,
            0x9e1a975b7f22d5ea,
            0xb5cc6ed68b2c108f,
            0x0a0cbaf933571e7a,
        ])),
    ],
];
