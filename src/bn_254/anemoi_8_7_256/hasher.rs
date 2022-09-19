//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{apply_permutation, DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
use super::{Jive, Sponge};

use super::Felt;
use super::{One, Zero};

use ark_ff::FromBytes;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// An Anemoi hash instantiation
pub struct AnemoiHash {
    state: [Felt; STATE_WIDTH],
    idx: usize,
}

impl Default for AnemoiHash {
    fn default() -> Self {
        Self {
            state: [Felt::zero(); STATE_WIDTH],
            idx: 0,
        }
    }
}

impl Sponge<Felt> for AnemoiHash {
    type Digest = AnemoiDigest;

    fn hash(bytes: &[u8]) -> Self::Digest {
        // Compute the number of field elements required to represent this
        // sequence of bytes.
        let num_elements = if bytes.len() % 31 == 0 {
            bytes.len() / 31
        } else {
            bytes.len() / 31 + 1
        };

        let sigma = if num_elements % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 31-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut i = 0;
        let mut num_hashed = 0;
        let mut buf = [0u8; 32];
        for chunk in bytes.chunks(31) {
            if num_hashed + i < num_elements - 1 {
                buf[..31].copy_from_slice(chunk);
            } else {
                // The last chunk may be smaller than the others, which requires a special handling.
                // In this case, we also append a byte set to 1 to the end of the string, padding the
                // sequence in a way that adding additional trailing zeros will yield a different hash.
                let chunk_len = chunk.len();
                buf = [0u8; 32];
                buf[..chunk_len].copy_from_slice(chunk);
                // [Different to paper]: We pad the last chunk with 1 to prevent length extension attack.
                if chunk_len < 31 {
                    buf[chunk_len] = 1;
                }
            }

            // Convert the bytes into a field element and absorb it into the rate portion of the
            // state. An Anemoi permutation is applied to the internal state if all the the rate
            // registers have been filled with additional values. We then reset the insertion index.
            state[i] += Felt::read(&buf[..]).unwrap();
            i += 1;
            if i % RATE_WIDTH == 0 {
                apply_permutation(&mut state);
                i = 0;
                num_hashed += RATE_WIDTH;
            }
        }

        // We then add sigma to the last register of the capacity.
        state[STATE_WIDTH - 1] += sigma;

        // If the message length is not a multiple of RATE_WIDTH, we append 1 to the rate cell
        // next to the one where we previously appended the last message element. This is
        // guaranted to be in the rate registers (i.e. to not require an extra permutation before
        // adding this constant) if sigma is equal to zero. We then apply a final Anemoi permutation
        // to the whole state. We then apply a final Anemoi permutation
        // to the whole state.
        if sigma.is_zero() {
            state[i] += Felt::one();
            apply_permutation(&mut state);
        }

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        AnemoiDigest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn hash_field(elems: &[Felt]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        let sigma = if elems.len() % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        let mut i = 0;
        for &element in elems.iter() {
            state[i] += element;
            i += 1;
            if i % RATE_WIDTH == 0 {
                apply_permutation(&mut state);
                i = 0;
            }
        }

        // We then add sigma to the last register of the capacity.
        state[STATE_WIDTH - 1] += sigma;

        // If the message length is not a multiple of RATE_WIDTH, we append 1 to the rate cell
        // next to the one where we previously appended the last message element. This is
        // guaranted to be in the rate registers (i.e. to not require an extra permutation before
        // adding this constant) if sigma is equal to zero. We then apply a final Anemoi permutation
        // to the whole state.
        if sigma.is_zero() {
            state[i] += Felt::one();
            apply_permutation(&mut state);
        }

        // Squeezing phase

        AnemoiDigest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiHash {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.try_into().unwrap();
        apply_permutation(&mut state);

        let mut result = [Felt::zero(); NUM_COLUMNS];
        for (i, r) in result.iter_mut().enumerate() {
            *r = elems[i] + elems[i + NUM_COLUMNS] + state[i] + state[i + NUM_COLUMNS];
        }

        result.to_vec()
    }

    fn compress_k(elems: &[Felt], k: usize) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);
        assert!(STATE_WIDTH % k == 0);
        // We can output as few as 2 elements while
        // maintaining the targeted security level.
        assert!(k <= NUM_COLUMNS);

        let mut state = elems.try_into().unwrap();
        apply_permutation(&mut state);

        let mut result = vec![Felt::zero(); STATE_WIDTH / k];
        let c = result.len();
        for (i, r) in result.iter_mut().enumerate() {
            for j in 0..k {
                *r += elems[i + c * j] + state[i + c * j];
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::super::BigInteger256;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger256([
                0x92e6d6661f44816e,
                0x006956f061ab59a4,
                0x3947d55c48185a9a,
                0x2175980110e12072,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0xb025916d24bf2833,
                    0xae3870c32547484d,
                    0x6de7ede95a4eb77a,
                    0x16684b487d8f9e6d,
                ])),
                Felt::new(BigInteger256([
                    0x7ef767558b492aa2,
                    0x4e810fd5db1cdf15,
                    0x40758e80248f0577,
                    0x0d99b997f7590cb6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2128a307bc0f2449,
                    0x559758e417fe6d74,
                    0x059ad5299921a4a6,
                    0x110fb3ac6df97d19,
                ])),
                Felt::new(BigInteger256([
                    0x9f692224e340b18c,
                    0x5cb10cc71bd7c298,
                    0x609a385ffd49cbc5,
                    0x007cf26650b00050,
                ])),
                Felt::new(BigInteger256([
                    0xb374442e1ef36c49,
                    0xdbeb9be306209f12,
                    0x9bfd63ecc167ac62,
                    0x15ad8c087bbf0868,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe50ecd663af5b6e3,
                    0xd14a8cfa56d421aa,
                    0xcf56b1786b14147e,
                    0x07676ca86b126033,
                ])),
                Felt::new(BigInteger256([
                    0x2ef095f859014c03,
                    0xd5acddbba327cdf1,
                    0xf7084bf87d0275ba,
                    0x2e70ea5ccf212192,
                ])),
                Felt::new(BigInteger256([
                    0xb8dbd4704015bd2e,
                    0xdcf452907fd9eda8,
                    0x3c6c4c9d5a650ca7,
                    0x12713576323209d8,
                ])),
                Felt::new(BigInteger256([
                    0xf26d5b9c64db6245,
                    0x7a9ea01289ddbcf8,
                    0x9cdf3dc954e52cb5,
                    0x29bd3d10784f44c1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x766d6be2676e513d,
                    0xe105c9beea2d96ba,
                    0xb42b73863c07e2cc,
                    0x1b19df129167c10a,
                ])),
                Felt::new(BigInteger256([
                    0xa49e1c42465df4b3,
                    0x5bd7d4bdfca44a6f,
                    0x7b17fe2eddb8ae17,
                    0x2b5f470163f13972,
                ])),
                Felt::new(BigInteger256([
                    0x74b40eebd4ae25ea,
                    0x2d45fc327ed09193,
                    0x19d9dc983ca17f48,
                    0x2c61280c9fc7938c,
                ])),
                Felt::new(BigInteger256([
                    0x64250b82afda09d9,
                    0x9e7e445763494899,
                    0x71bb7ddbdddc9054,
                    0x2e2b608a794ed75e,
                ])),
                Felt::new(BigInteger256([
                    0xd67f39333690eab6,
                    0x8682a2664be8da26,
                    0xce6aff0dbf2e33af,
                    0x0fc518bc2d6f42a6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5fb3c338bf124909,
                    0x0ad4a9ae69dff9a1,
                    0x4562d9c91515d3da,
                    0x0d6458175e97564d,
                ])),
                Felt::new(BigInteger256([
                    0xb9d36263e9ae38fe,
                    0x1660b8ba090257ea,
                    0x2ac2894c945b6b13,
                    0x14e043612a4cf90b,
                ])),
                Felt::new(BigInteger256([
                    0x51e6030c60b4f6cd,
                    0xc823a1c4cadd1b59,
                    0xd3cff40f6ce676fc,
                    0x28987f18f1898d42,
                ])),
                Felt::new(BigInteger256([
                    0x03db25dacd842973,
                    0x16ee3c032b7ff53c,
                    0x331164883082daf6,
                    0x0ee868cb5b94ea64,
                ])),
                Felt::new(BigInteger256([
                    0x6952832131abd4be,
                    0xe0b90aea131afcad,
                    0x147e1afcf64efaf8,
                    0x2ef25d773654c17a,
                ])),
                Felt::new(BigInteger256([
                    0xb3d6c9385f76baa2,
                    0x29848b3819958e72,
                    0x9eb052bb9efd480d,
                    0x23bf0b56f0323492,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x5b81d8ee69385a99,
                    0x3b1670210e3b87cf,
                    0x68d814035003845e,
                    0x0e5ee9b35d62a9ca,
                ])),
                Felt::new(BigInteger256([
                    0x2be5e8c8320abdc0,
                    0x0f812f83e67286f6,
                    0x7854730098c47f43,
                    0x071f41dcab59546d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x383e102dbd1e7a2e,
                    0xa7148dbeb0cacecd,
                    0x77996a7439812826,
                    0x2c86217472ab4efc,
                ])),
                Felt::new(BigInteger256([
                    0xfbacbb9cb3691609,
                    0xc48aca5cd75be355,
                    0xd2398545ff93e9d5,
                    0x139d220ee53e8c9e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9b0cce73a4b6326b,
                    0x0de0ea422ac1894b,
                    0xda6406cef03d11a1,
                    0x1a573c6233acd6eb,
                ])),
                Felt::new(BigInteger256([
                    0x87c4b7d621cbdef2,
                    0x819cc54e7d116c70,
                    0x7ca7c6044c7561d6,
                    0x27e6d719155cfb65,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x206212a6bc8c3ca6,
                    0x3712cdf8c2c454f6,
                    0xcb4bfa88aafba8d9,
                    0x2b44347a8384c40d,
                ])),
                Felt::new(BigInteger256([
                    0x8d489e303337e0b3,
                    0x4f10dd584823476d,
                    0xe129a3362a381a1c,
                    0x1bb3adc1cf21e019,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x423ffa725d665e0c,
                    0x704b3507a2e04ef3,
                    0x19e1b50008508aa5,
                    0x0e6c9bcbc78b8f8a,
                ])),
                Felt::new(BigInteger256([
                    0x07119a18e8d68a22,
                    0x881d0f89e94d161c,
                    0x1911bc3aa3a3f983,
                    0x19d03f192e10a647,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xec30abe7e851a97e,
                    0xe70c76c772be7e78,
                    0x6e0470beaf21a86d,
                    0x1c566142d95d2474,
                ])),
                Felt::new(BigInteger256([
                    0x854ab80ab05f605b,
                    0x75684c3249c80231,
                    0x900f4980e72bdae8,
                    0x1d99f175988e08e5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x77eb63e9fa1a960d,
                    0xa9d0b76fcbe2c8a2,
                    0xb9a5bb745a6a83d9,
                    0x1ad9a91f28579567,
                ])),
                Felt::new(BigInteger256([
                    0x6027ebbbae440217,
                    0xd5f6998982a8bdb1,
                    0xf7409814ab73ada9,
                    0x2909cdc872857406,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x291c531d60f5906f,
                    0x9e3e9c8a2991b683,
                    0x4372a3665b683a0a,
                    0x020fb10232384d7d,
                ])),
                Felt::new(BigInteger256([
                    0xc772f34696fb8626,
                    0xdfe287d1faac875c,
                    0xe3c3b10df3a194c7,
                    0x16f7edcd92038aa2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x67d0832c57c1b788,
                    0xdb2623a4c9d02802,
                    0x471f3bbe007bc1bf,
                    0x104f013a9a1e418b,
                ])),
                Felt::new(BigInteger256([
                    0xf714be8bbb398bd0,
                    0xc0c114e919d76120,
                    0xe6275a66579e0a8a,
                    0x0955507a9451cbed,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x053470bcb2bf4a9c,
                    0xdc2dfc87d3c19060,
                    0x6b382aebdf20a420,
                    0x11d81fefe59ca5f8,
                ])),
                Felt::new(BigInteger256([
                    0xe22783d2d42b63fd,
                    0xba72c0717470448b,
                    0xed5de31ff031d8dd,
                    0x252d92869d3a9d74,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 8],
            vec![Felt::one(); 8],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x1e2e16a4d49c45c5,
                    0xce351bb05ed7d3d8,
                    0xf373e2632ae68aa0,
                    0x050d3327c879d3c9,
                ])),
                Felt::new(BigInteger256([
                    0x6fd984f8842b0924,
                    0x046a1be6bc3c2f58,
                    0xe910d638a51ae22b,
                    0x0001a169e7752ee9,
                ])),
                Felt::new(BigInteger256([
                    0x8e69d472f8ee3902,
                    0x0bdcffcfe770658a,
                    0x72391e7b9aec2247,
                    0x252d88c65abe32fd,
                ])),
                Felt::new(BigInteger256([
                    0x173f3d45bd20295b,
                    0x7dfdbe64edfddbc1,
                    0xd10ecfdd6ac768c4,
                    0x0f66113b5bf7ca72,
                ])),
                Felt::new(BigInteger256([
                    0xf0b3a01d432bb3a4,
                    0xe1ee55c36ce84f27,
                    0x1db8d17e6d117e04,
                    0x18cf3c68e305f6d1,
                ])),
                Felt::new(BigInteger256([
                    0x2a6db55b0990b162,
                    0x948104c247863939,
                    0x78f83a7b50b7fe13,
                    0x2f6007107dd52351,
                ])),
                Felt::new(BigInteger256([
                    0x5a6797cc1e86a3a0,
                    0xecb2613013204470,
                    0x44238ae1d08cb20f,
                    0x06f2d725c7aa704f,
                ])),
                Felt::new(BigInteger256([
                    0xadc79abe622875d6,
                    0x4146403d7c074eb0,
                    0x59c067d3db996d4e,
                    0x14e1a5fb288d1c6c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x830496671ad0ceac,
                    0xf4c81541fbc82537,
                    0x54c6d9b2d877d6a5,
                    0x0a98a25d603b1760,
                ])),
                Felt::new(BigInteger256([
                    0xd7c89f9b571fe2e3,
                    0x5b130aeee0dfe078,
                    0x33ce5f6611c7e206,
                    0x01582190c0c486b5,
                ])),
                Felt::new(BigInteger256([
                    0xd7374426246c43ed,
                    0x35b23069ca19ac7e,
                    0xeabc56adaffb39f5,
                    0x127adc42be9991a7,
                ])),
                Felt::new(BigInteger256([
                    0x3dd060f31c9aa0e3,
                    0x9be19ec60705302e,
                    0x6527a86c479e91f4,
                    0x0d2154f06e9e8e3b,
                ])),
                Felt::new(BigInteger256([
                    0x6c61c1f2bd7246a7,
                    0x3328615afff8f7f3,
                    0x0fdf260df0be896e,
                    0x06d958dca9cbe328,
                ])),
                Felt::new(BigInteger256([
                    0x1112ea4946e63fce,
                    0x5a9807d12fa450c0,
                    0xc2257034fc93cbb3,
                    0x19fc0cb4cd728a9c,
                ])),
                Felt::new(BigInteger256([
                    0x735855f04b8d058d,
                    0x3ea29f42983098a2,
                    0xe1ac71410bbfc43d,
                    0x12560931fab0cdb2,
                ])),
                Felt::new(BigInteger256([
                    0xd0597a58fdd8b8d6,
                    0x8668f3a790339950,
                    0x0595689af77b6a19,
                    0x2fa04025a8a7c6ee,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x70ea74693a14456f,
                    0x4ee9457d30f4c40d,
                    0xbb8ccfc9cb6a24e0,
                    0x28eb2fa977111ad2,
                ])),
                Felt::new(BigInteger256([
                    0x88a3c59dd7f5e68c,
                    0x1c15bc727befc12f,
                    0xb5441b6f1abb93d2,
                    0x22d3165e868eb253,
                ])),
                Felt::new(BigInteger256([
                    0x939f205b5ef9fb92,
                    0xef004bc2a73e533e,
                    0xed89b0c971727cbe,
                    0x0be81a956c795288,
                ])),
                Felt::new(BigInteger256([
                    0x49f179a66a4615f8,
                    0x8dbf7a73481e6d46,
                    0xa65396617358a7ff,
                    0x0a35621ad8d01f79,
                ])),
                Felt::new(BigInteger256([
                    0xf626c3bac5c4685b,
                    0x2436f1a6bbf457e0,
                    0x4e960890ef9213f2,
                    0x1dca3353c1979215,
                ])),
                Felt::new(BigInteger256([
                    0x65606b6456268906,
                    0x3f55e090cea270df,
                    0x6442a51f0cbd1c17,
                    0x25e6577244d65f61,
                ])),
                Felt::new(BigInteger256([
                    0xf51fa687e0098da5,
                    0x69ef0a9926e6d48b,
                    0x0240bc228d57cd96,
                    0x145619c6085afa69,
                ])),
                Felt::new(BigInteger256([
                    0x42fbe76926f613e8,
                    0x0317b3b279d98b03,
                    0xf2c6c002ee4efd9e,
                    0x11f6833f952a4956,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xfa98d6e8be8493be,
                    0xaa23528abf17fbde,
                    0xc267e06c24ad58e0,
                    0x17dae2f81623236a,
                ])),
                Felt::new(BigInteger256([
                    0x5aa445e2248d0d38,
                    0xa3805d218e076f05,
                    0x3b56ff5a9be36ff5,
                    0x020ee6654834720c,
                ])),
                Felt::new(BigInteger256([
                    0xc4ede5c4f2383d70,
                    0x0676c557d8f90939,
                    0x5f3895281300a910,
                    0x04d401109f5e112d,
                ])),
                Felt::new(BigInteger256([
                    0xb1130e7959e57bd2,
                    0xda7e62ede2dee978,
                    0x848caac19dbbb1dd,
                    0x04636366213e0bf8,
                ])),
                Felt::new(BigInteger256([
                    0x6d9c0062b5df5a3f,
                    0x2fe7ed0bb0fcc3ee,
                    0x0c08d0dc20598515,
                    0x194c66d1e8b7862f,
                ])),
                Felt::new(BigInteger256([
                    0xa7cc2343c0b97566,
                    0x5ab5c237f8ef3261,
                    0xe3a716ffaf5a30ff,
                    0x06b1f8412fe5f0ed,
                ])),
                Felt::new(BigInteger256([
                    0x87184e12f38feb7d,
                    0xcd2e9a5c80e870f3,
                    0x6d13ffe973b74ece,
                    0x12633e7e46d49d97,
                ])),
                Felt::new(BigInteger256([
                    0x53685bb8ebc91f7c,
                    0x05724fe1c54a3378,
                    0x9690959d09eac32e,
                    0x15c18fae4b87374c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x99185b15a40d6da5,
                    0x630a70156bde4f25,
                    0x0b8068d4624acdd4,
                    0x129d6956caefb6f8,
                ])),
                Felt::new(BigInteger256([
                    0x6c9598c98aee4335,
                    0x3ff284f2abe0a15d,
                    0xe804e8923b15fdd8,
                    0x250346c03b4140af,
                ])),
                Felt::new(BigInteger256([
                    0x6ff47fc028085d0b,
                    0x2ea6af41ea590865,
                    0x27899aefea858a2b,
                    0x25e2a8c9f671d180,
                ])),
                Felt::new(BigInteger256([
                    0x04221f9199050425,
                    0x1a085c7891b1e8eb,
                    0x48980d632b7fdaa6,
                    0x0e201509ca9cbc65,
                ])),
                Felt::new(BigInteger256([
                    0xfb62a0b49503a85b,
                    0x1c2e382b3b512670,
                    0x902204aaa39e1ab1,
                    0x220b0a9fb988d06b,
                ])),
                Felt::new(BigInteger256([
                    0x963fd6f2515d0b4f,
                    0xb62bcc5d6d4e02dc,
                    0x8a4164d7c6d76d01,
                    0x054b1da2f0ca760b,
                ])),
                Felt::new(BigInteger256([
                    0x9813bdb23b857e7a,
                    0xc2f97630efd12f0e,
                    0x806e17927a673703,
                    0x0e409b6f816e813c,
                ])),
                Felt::new(BigInteger256([
                    0xf4a2e2d82f8678d3,
                    0xb0c2515655da9c28,
                    0x911860a4cfd25de3,
                    0x2d00c1ebb690ee8f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x57a55202466b603f,
                    0x178f6473083ad507,
                    0x13cb279c751045e6,
                    0x1a76dd437fe4e181,
                ])),
                Felt::new(BigInteger256([
                    0x0ff11554220f3c54,
                    0x9304e09812d748b5,
                    0x94edc4e004a1e0dd,
                    0x05f48ed4b3ebb1c2,
                ])),
                Felt::new(BigInteger256([
                    0x04d91ba94646d6bd,
                    0xcdb45bfd33473eaa,
                    0xac878eade05faedf,
                    0x25282151cc098039,
                ])),
                Felt::new(BigInteger256([
                    0xb931f27f063617ca,
                    0x0bab50b8c80d15cf,
                    0x136adf40048b26a5,
                    0x2111273a7075982a,
                ])),
                Felt::new(BigInteger256([
                    0x5daf09194cc2c205,
                    0x360ed1ee30ea64eb,
                    0xa2f9c077be5057ac,
                    0x1fb8b86861adc6b4,
                ])),
                Felt::new(BigInteger256([
                    0x1216fc607046dab8,
                    0x8c68e1ef2858a79e,
                    0x380637d6209ad6ea,
                    0x0990ca0f01c63efa,
                ])),
                Felt::new(BigInteger256([
                    0x9e39c16d3e0ed0d8,
                    0x62bd3ca78d929ae3,
                    0x166231a794290ad6,
                    0x2d8eb4df7d595551,
                ])),
                Felt::new(BigInteger256([
                    0x5da4a5f0ab2eb7f4,
                    0xddf8974826e1b796,
                    0xf28b7739670e6c22,
                    0x00cd13ba9393f887,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xaa2ffb65e026e291,
                    0x4f2206b84f20007b,
                    0x8b45301af7533224,
                    0x2e7730ad66e0d887,
                ])),
                Felt::new(BigInteger256([
                    0xfd20f6c946e179a8,
                    0x5be87317666113d2,
                    0x0baa39d0b3939b90,
                    0x265878677161758c,
                ])),
                Felt::new(BigInteger256([
                    0x72f8e3d349fcaf1a,
                    0x28592a3c282cd03c,
                    0x8e2025364f40eeea,
                    0x0975a7cb7d83696c,
                ])),
                Felt::new(BigInteger256([
                    0xa2719230cc152cb0,
                    0x2a1a35f6935d2a38,
                    0xf10224b63c91c500,
                    0x2b6f002a097da121,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc262702c65fd0ebd,
                    0x10a4222e2208b81c,
                    0xf62b1351722169d8,
                    0x1f06b385dec0c17a,
                ])),
                Felt::new(BigInteger256([
                    0xa4cbef0c14e267ff,
                    0x9f5515bbd2c8d785,
                    0x76da2a6d2eb87c95,
                    0x1e283a1dbb595ecb,
                ])),
                Felt::new(BigInteger256([
                    0x9eebd26a2e6e4bcd,
                    0x17c09d32e6ef03bd,
                    0xaf38e4dc4269f48f,
                    0x00cac09f3596b67b,
                ])),
                Felt::new(BigInteger256([
                    0x9dd1e3b95864cb60,
                    0xdfd8adafa833a46a,
                    0x94c2f6a453bdefca,
                    0x0373db1114988ba2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf688b4acc2f6f7b3,
                    0x950d970de970d8a6,
                    0x8afe10846bfe72ea,
                    0x125b29ced6a4990f,
                ])),
                Felt::new(BigInteger256([
                    0x181ae446e9da5f1b,
                    0x3ef2a71be91721d1,
                    0xcd9331d17a68feae,
                    0x0b49ee707da7eba4,
                ])),
                Felt::new(BigInteger256([
                    0x952faf7ce53b5652,
                    0x42f6c3ae9b4a287b,
                    0xb70612ce1ddc71ec,
                    0x170ae0d4f7019d28,
                ])),
                Felt::new(BigInteger256([
                    0xd18ead97b45a468f,
                    0xfbac13ff39046fd5,
                    0xe5de2a5ce7360a3c,
                    0x0d4f1643f8ccfa8c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb5dd77e85fe132a7,
                    0x1bef161c8e184e54,
                    0x3fe9f0dc16f94747,
                    0x23e29f4c63678cf3,
                ])),
                Felt::new(BigInteger256([
                    0x28e98f0002a444f3,
                    0x1948a8b600fe4442,
                    0x54edad9c2114bb45,
                    0x044cf3a37e042256,
                ])),
                Felt::new(BigInteger256([
                    0xd1bd7b87efa7b719,
                    0x66d2ebaf4b9ea584,
                    0x08ea1b7d3d631790,
                    0x11747566777d59bf,
                ])),
                Felt::new(BigInteger256([
                    0x453f9d77a2cccf88,
                    0xe7841383d1ec08c7,
                    0xd1ff940061661c24,
                    0x24903de5ceb4a2ea,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5a5c72129f568283,
                    0xa6712789e929168d,
                    0xf2ebc3fa9a727ac3,
                    0x22b5fb7a4a44b1dd,
                ])),
                Felt::new(BigInteger256([
                    0xff1e43c994f65f86,
                    0x14216a61ced65e88,
                    0x59f4b41b469ab7a8,
                    0x1b7947aa93eb6fe4,
                ])),
                Felt::new(BigInteger256([
                    0x52b2f00c48a0183a,
                    0xffc7e2a8da7f959b,
                    0x94d0432a17d2d0f9,
                    0x0082046d03e6f09d,
                ])),
                Felt::new(BigInteger256([
                    0xfc5dc5f1d4d9ff62,
                    0xa5c7f33d5a034b99,
                    0x2f8070bb3d3fd521,
                    0x06e3e528f1a59c3a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd5d3d8a015af86ef,
                    0x0d21e3015309593a,
                    0x38bcd76b5aee86c4,
                    0x1144fedf651f39dd,
                ])),
                Felt::new(BigInteger256([
                    0x29de0e9217553518,
                    0x985cba9f2dc83b6e,
                    0x2d7a5bd3fa31dcd2,
                    0x2f32068b237e17bc,
                ])),
                Felt::new(BigInteger256([
                    0x2921493675c277d6,
                    0xa2379856f9b6735f,
                    0x6325a9ac8343f703,
                    0x14617a87ecc7ec85,
                ])),
                Felt::new(BigInteger256([
                    0xb49deac3e6bea664,
                    0x78ac177f03de335a,
                    0x2e3d81cda518c0c2,
                    0x1136af880f1a5d07,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x78d50cec4b71ec7d,
                    0x3949618adc6b88d5,
                    0x2a373d967aa942e4,
                    0x2749c4c75af617d3,
                ])),
                Felt::new(BigInteger256([
                    0x1444ce11954dff07,
                    0x7969bdcd90ada728,
                    0x6cd6c49bc1e81d26,
                    0x2b7bd555d12dc478,
                ])),
                Felt::new(BigInteger256([
                    0xa598bfda4383ad13,
                    0xa8a17ab3f8a1882f,
                    0x91e01564769f247f,
                    0x112810945cdbf47e,
                ])),
                Felt::new(BigInteger256([
                    0x519a844efadad3c2,
                    0xe855fca76c2d66d4,
                    0xe1ef85bc7ca25d8d,
                    0x0a783a1b3d7d33aa,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbcb15e4aa09493b9,
                    0x2c3f892a7cbcb773,
                    0x8e31b28b01db809a,
                    0x2a8db0f41db16514,
                ])),
                Felt::new(BigInteger256([
                    0xc1302522f64d1354,
                    0xcdcb608e668e61cb,
                    0x685c78c64beb4083,
                    0x2ab0eab8fd92eee9,
                ])),
                Felt::new(BigInteger256([
                    0xe5ce93d1ebcda421,
                    0xafcaa9a026ee351b,
                    0x0af331f5413aed7d,
                    0x1dd701d6f78e367d,
                ])),
                Felt::new(BigInteger256([
                    0x08285739e532fccb,
                    0xbd00d956305d8ab3,
                    0xbf8b5101f2f2eb47,
                    0x1e0862c88ab8e5e5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9097fbe192a59bed,
                    0xbcf88f3141f67564,
                    0x512c6f4736416a4e,
                    0x148820685f5623e7,
                ])),
                Felt::new(BigInteger256([
                    0xaff1a105dd1d1f1c,
                    0xa96a40f05ce26ff7,
                    0x75f08f355687de2f,
                    0x0db32251d87a3632,
                ])),
                Felt::new(BigInteger256([
                    0xdbfaaeca85538ff7,
                    0xcf1ff4152da5a9a1,
                    0xaea5ba19e19048aa,
                    0x1967adfdb3f8a366,
                ])),
                Felt::new(BigInteger256([
                    0x696a2d589878bec2,
                    0x1671279cd2368471,
                    0xd11cd598e26e0fb4,
                    0x13dfcbe9f9c857a7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xadf32760d1f7b441,
                    0x3923b9a0cd82437e,
                    0x8e029b010ffe2151,
                    0x1462ec5c52a9d20f,
                ])),
                Felt::new(BigInteger256([
                    0x66ff20dc7eecb747,
                    0xa10cfa142d84f7f2,
                    0x76e6063e11a3c5ae,
                    0x22d035640e380989,
                ])),
                Felt::new(BigInteger256([
                    0x4a369cbc212ea8a1,
                    0x798b476af858d0d5,
                    0x1c45298b16f555a1,
                    0x2228f2ce671a3fa7,
                ])),
                Felt::new(BigInteger256([
                    0x99e9d8da591ebcf7,
                    0x31379aa523fdb3dc,
                    0x39189756bc93d489,
                    0x20c7a18f6eb666e4,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 2));
        }

        let input_data = [
            vec![Felt::zero(); 8],
            vec![Felt::one(); 8],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2fe8157477496ee4,
                    0x3fd18a9de001d431,
                    0x2dc0ce2f3547b40a,
                    0x173c70b841d107aa,
                ])),
                Felt::new(BigInteger256([
                    0x55b6af51b9718b10,
                    0x434bbe0a850ecf21,
                    0x99409bc5c95ffa8b,
                    0x1544d0fe6d916360,
                ])),
                Felt::new(BigInteger256([
                    0x5b25c3ab67bea0b3,
                    0xc8713e80258617ea,
                    0x65b342dd116e9c39,
                    0x070c1fe62c5df30f,
                ])),
                Felt::new(BigInteger256([
                    0xf589ca549f6e0b7f,
                    0x7563eb85753aaa8f,
                    0x7b7120fce9684405,
                    0x0fd1cb8597bc0df1,
                ])),
                Felt::new(BigInteger256([
                    0xe0278bc6a194bdaa,
                    0x63bf20dde1d4500a,
                    0xea2efb22c89c2654,
                    0x16bd6ef56da2c78b,
                ])),
                Felt::new(BigInteger256([
                    0xe6be39178affd5fd,
                    0x4361d8d56ff6fc80,
                    0x819e9d27e6fb3a5a,
                    0x261c87eff894842e,
                ])),
                Felt::new(BigInteger256([
                    0x9b48624ad0922b9f,
                    0x6203ac37370bd344,
                    0xa18b14c8c0e2eb50,
                    0x2995d71b77d1559f,
                ])),
                Felt::new(BigInteger256([
                    0x279f3387158648ba,
                    0xeddd8421cc9a2236,
                    0x714c6313b9f22859,
                    0x010a2e731ae67991,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x9df8d4018cfcc31f,
                    0xcd06f323c855b503,
                    0x706206382c3570fa,
                    0x244dacab32c2aa24,
                ])),
                Felt::new(BigInteger256([
                    0x9e7ee77a22d1c271,
                    0xbfd760065c5c7f08,
                    0xd728f63ecf694caa,
                    0x302f47cfa9d6fafa,
                ])),
                Felt::new(BigInteger256([
                    0x83783e8d88f9fa8a,
                    0xdb4ba8ac0885d96a,
                    0x6ac5fa9c7fe9afb5,
                    0x132d8469d67cf742,
                ])),
                Felt::new(BigInteger256([
                    0xe407779dd0d20172,
                    0xe0bfe377d6939334,
                    0x279691ac0602ed10,
                    0x1a4d5a9f51dc903a,
                ])),
                Felt::new(BigInteger256([
                    0x81f910682aea865f,
                    0xb73fd54453accb85,
                    0xc426b00b009ffb64,
                    0x151c611047024078,
                ])),
                Felt::new(BigInteger256([
                    0x4bc0e3cb71ee2088,
                    0x74a1db83aaaeb9de,
                    0xf59b502d408c12b0,
                    0x0ca9b7d55584f802,
                ])),
                Felt::new(BigInteger256([
                    0xac5bc6f2971ffaea,
                    0x55ea3e95c039f1d3,
                    0xdc1547581f77b1f0,
                    0x226d088ccb4f9c7c,
                ])),
                Felt::new(BigInteger256([
                    0x8583f0ae429db0d7,
                    0x822b5c0513ce96e8,
                    0xff21a5b721ab2212,
                    0x20ee3183926f3c09,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x30bdd8cd7b0c37a1,
                    0x9ffe4ec37839397c,
                    0x6934d5e6568737ef,
                    0x137fde375ca815a9,
                ])),
                Felt::new(BigInteger256([
                    0x73b95cbdea6ab655,
                    0xae5cd3a0d9dd5e36,
                    0x5de06faa64ff1fa1,
                    0x228a3aa29f61595a,
                ])),
                Felt::new(BigInteger256([
                    0x1062714c4b1d4922,
                    0xdd4208aba0745d14,
                    0x10e546c2b6e08747,
                    0x10bc3d7e065596df,
                ])),
                Felt::new(BigInteger256([
                    0x1626616b374ea8ef,
                    0xc838a4385e728c36,
                    0x43c1fa7076111337,
                    0x16f8386d6116f3bc,
                ])),
                Felt::new(BigInteger256([
                    0xcfedf57ac3eb654f,
                    0xcd65f6175ea8e1d5,
                    0x501de0a4036bd18b,
                    0x07d056f43e9bb2ff,
                ])),
                Felt::new(BigInteger256([
                    0x5191542b686b0241,
                    0xf037441d31ef4499,
                    0x485e9b3f7ac678ce,
                    0x0c181276851a5201,
                ])),
                Felt::new(BigInteger256([
                    0x10f03fd146c5f835,
                    0xfec837fee44e63c4,
                    0xee85f0af368ae4e6,
                    0x0e286d9f80af7deb,
                ])),
                Felt::new(BigInteger256([
                    0x8cbef12c5503c658,
                    0xba611e07b0e759f9,
                    0xda41f6062e168abd,
                    0x0bea103d8fa9338d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2bcd02e8f1f5dc08,
                    0x2261891f63e7500e,
                    0x471176c764d5f0d5,
                    0x2d9f051d3913492b,
                ])),
                Felt::new(BigInteger256([
                    0xa7c8c0d17844d1ed,
                    0x4b881326d71f85d2,
                    0xd0842d05a8e9893c,
                    0x195a7afc0a2be612,
                ])),
                Felt::new(BigInteger256([
                    0x74594bcfc1eb4a02,
                    0x11003183bf6a5e8c,
                    0x172e65808c4b1b82,
                    0x2ad208c71f8e5130,
                ])),
                Felt::new(BigInteger256([
                    0xf9fd967c8f9016cc,
                    0x475c95dba03fc58f,
                    0x573f9db7027778b3,
                    0x09c133444f56aa53,
                ])),
                Felt::new(BigInteger256([
                    0xd8070accc03f4261,
                    0xccfcb9d8e5aa3206,
                    0x818f1de7d8fef75e,
                    0x2fcedd5bcd3bb4b3,
                ])),
                Felt::new(BigInteger256([
                    0xe928b2dafa3406f2,
                    0xc60c85988231213b,
                    0x2ef9c6526bea3fda,
                    0x03ad1c663e6f3dd0,
                ])),
                Felt::new(BigInteger256([
                    0x3dbeb306d29a9337,
                    0x042a36d7f5817491,
                    0x6c4e2965643bd93f,
                    0x23f7665487f861d3,
                ])),
                Felt::new(BigInteger256([
                    0xdb58774ae5052a64,
                    0xf5c57367ca3c3687,
                    0x467879f258003492,
                    0x15abfb8b355769c6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x4354762c23959b77,
                    0x73c1bc56f7e37804,
                    0x47b093fdf5029709,
                    0x185cfe1d5eb6421b,
                ])),
                Felt::new(BigInteger256([
                    0xe46b2c3fad4f2baa,
                    0x3b6a1dd4561daa63,
                    0xdada3c00c4c3edfc,
                    0x0d434727369f77bd,
                ])),
                Felt::new(BigInteger256([
                    0xc4644fe6f5e7c857,
                    0x9a29e9e2ca29aeec,
                    0x418bbbc355e4161b,
                    0x1d302fa47616db65,
                ])),
                Felt::new(BigInteger256([
                    0xb4b8a759fbe0bc52,
                    0x659714668b12291d,
                    0xf4b2e8cc548bb6a2,
                    0x1d6f70505ecafac8,
                ])),
                Felt::new(BigInteger256([
                    0x1b27656000419358,
                    0x7888db889559e180,
                    0x28bbf5803348d085,
                    0x20fdc7ad4a9c62c3,
                ])),
                Felt::new(BigInteger256([
                    0x29f1fbdc3723ee92,
                    0x280006cb06b8b493,
                    0x98a71a9151e52744,
                    0x138e5a386c961c36,
                ])),
                Felt::new(BigInteger256([
                    0xd39290509f8ca179,
                    0x6aec4da7ac195b96,
                    0xbf6cbb8713425890,
                    0x129e0ae20be8c249,
                ])),
                Felt::new(BigInteger256([
                    0x7c14596d4206222a,
                    0x1035934a6bd10cae,
                    0xe13ec49944bb21b3,
                    0x04b8fe76ff477c00,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x65c391f4f436ecb3,
                    0xa70a22bb4b4f2e72,
                    0xb74d518d6c8a1055,
                    0x1c0d524d99c27546,
                ])),
                Felt::new(BigInteger256([
                    0x75019e42ecf38a16,
                    0xe7cd19f02f785264,
                    0xc2fb500bc461568a,
                    0x265db7179dc2b53f,
                ])),
                Felt::new(BigInteger256([
                    0x3cdcf0e46258e1d1,
                    0xbad5acd1128808b3,
                    0x612b3eb19fd71543,
                    0x190d85d3e03c81c0,
                ])),
                Felt::new(BigInteger256([
                    0x7c64160c6145b9c5,
                    0xd1c920192291bb75,
                    0xa6c3c1524f64b10c,
                    0x1af80326213245a4,
                ])),
                Felt::new(BigInteger256([
                    0xc42ef22bdc345763,
                    0xb2d1a28627ee8177,
                    0x84ceac5c931d7cd8,
                    0x2f7a9d05d96b6f1c,
                ])),
                Felt::new(BigInteger256([
                    0xeb05766ae55585f3,
                    0x76f6a0e50bd48859,
                    0x1032a8c7a5caccea,
                    0x134663c95a62a402,
                ])),
                Felt::new(BigInteger256([
                    0xa9e9d9a428b96a37,
                    0xfc90f01fc5790cb1,
                    0x411ca0d13a96838e,
                    0x08f8d3f33d352e3b,
                ])),
                Felt::new(BigInteger256([
                    0xbd0d37dadb3680d8,
                    0x7321e2d1cde4575e,
                    0x87963b4171e9e5f6,
                    0x2bcfc250d36673c3,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xe108532251a69464,
                    0xdff9c6630edb062a,
                    0x61150f9ac512c8b0,
                    0x07888a060332a1ca,
                ])),
                Felt::new(BigInteger256([
                    0x6371fce33a79a911,
                    0xee813e7c914c737e,
                    0x445c18d06ea40832,
                    0x21632a1e99ad7684,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x614e4296946b5a8a,
                    0x2864bf6108f7bbda,
                    0xa563f82db48b5e67,
                    0x1fd17425145777f6,
                ])),
                Felt::new(BigInteger256([
                    0x429dd2c56d47335f,
                    0x7f2dc36b7afc7bf0,
                    0x0b9d211182766c60,
                    0x219c152ecff1ea6e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8bb86429a8324e05,
                    0xd8045abc84bb0122,
                    0x4204235289dae4d6,
                    0x29660aa3cda63638,
                ])),
                Felt::new(BigInteger256([
                    0xe9a991de9e34a5aa,
                    0x3a9ebb1b221b91a6,
                    0xb3715c2e619f08eb,
                    0x189904b47674e631,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4b7a6759770bec79,
                    0xeb40973a7145294c,
                    0x9083c6a2d2db0679,
                    0x04f2c63ff9b34688,
                ])),
                Felt::new(BigInteger256([
                    0x6e292c77a571147b,
                    0x00ccbc39d2ea4d09,
                    0x26ed419c827ad76a,
                    0x28dd31894cb8c541,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9ff13f867102428d,
                    0x5a904b8865fa3a8e,
                    0x14eb1820c84cda9e,
                    0x194a29841a296a39,
                ])),
                Felt::new(BigInteger256([
                    0x6f8c4d626e4bb5d6,
                    0xa867386351e3e160,
                    0xe5b52ca762eeb94b,
                    0x036303239bef6cf6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x036cc84d9aa91e46,
                    0x9942cbb1669d9915,
                    0x986251f5771f04a4,
                    0x0b4d20328ad7d575,
                ])),
                Felt::new(BigInteger256([
                    0xf927db023bc50d51,
                    0xf4ba38eaf0af03b6,
                    0x8f5d98f210d87bfb,
                    0x1ee4a823cf0ee6f7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc7fdd0eaa479802c,
                    0xdfa9f9b52dc0b319,
                    0x41c7f35ba98858b3,
                    0x14fd43f7019bca53,
                ])),
                Felt::new(BigInteger256([
                    0x0bda5b4cfc25b943,
                    0x232ab3c7db0e1de5,
                    0x0d10729054f8ae8d,
                    0x0de91bb7c213fc02,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7b0cd48fb0e45266,
                    0x2144a33c433f3bb6,
                    0xa3314755b5981cbd,
                    0x19a0ce02dd15b6a5,
                ])),
                Felt::new(BigInteger256([
                    0x966400334eebcc56,
                    0x5e492287e6e105d1,
                    0xcc3098a1459cc8be,
                    0x1cfb08d3e5ba58c0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x74532cc4aacfcb27,
                    0x778507fbf429b524,
                    0xf0f486f63b78095a,
                    0x160422f60bd6f3e5,
                ])),
                Felt::new(BigInteger256([
                    0x07a6096635b2a38f,
                    0x29e01696b1c4a733,
                    0x6d3505ddbf47b0ed,
                    0x0700990262d728b8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7da81308583310b9,
                    0x8cacd9ee3642230a,
                    0x08e7d4cc002d7fd5,
                    0x02fe2c6b37892e34,
                ])),
                Felt::new(BigInteger256([
                    0xe98f288d4b1071b4,
                    0xbe01da8c821f48f4,
                    0xd072da244806d296,
                    0x091f4adb049b7e6e,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
