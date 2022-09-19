//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{apply_permutation, DIGEST_SIZE, STATE_WIDTH};
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
        let num_elements = if bytes.len() % 47 == 0 {
            bytes.len() / 47
        } else {
            bytes.len() / 47 + 1
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 47-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut buf = [0u8; 48];
        for (i, chunk) in bytes.chunks(47).enumerate() {
            if i < num_elements - 1 {
                buf[0..47].copy_from_slice(chunk);
            } else {
                // If we are dealing with the last chunk, it may be smaller than 47 bytes long, so
                // we need to handle it slightly differently. We also append a byte set to 1 to the
                // end of the string if needed. This pads the string in such a way that adding
                // trailing zeros results in a different hash.
                let chunk_len = chunk.len();
                buf = [0u8; 48];
                buf[..chunk_len].copy_from_slice(chunk);
                // [Different to paper]: We pad the last chunk with 1 to prevent length extension attack.
                if chunk_len < 47 {
                    buf[chunk_len] = 1;
                }
            }

            // Convert the bytes into a field element and absorb it into the rate portion of the
            // state. An Anemoi permutation is applied to the internal state if all the the rate
            // registers have been filled with additional values. We then reset the insertion index.
            state[0] += Felt::read(&buf[..]).unwrap();
            apply_permutation(&mut state);
        }
        state[STATE_WIDTH - 1] += Felt::one();

        // Squeezing phase

        let mut digest_array = [state[0]; DIGEST_SIZE];
        apply_permutation(&mut state);
        digest_array[1] = state[0];

        AnemoiDigest::new(digest_array)
    }

    fn hash_field(elems: &[Felt]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        for &element in elems.iter() {
            state[0] += element;
            apply_permutation(&mut state);
        }

        state[STATE_WIDTH - 1] += Felt::one();

        // Squeezing phase

        let mut digest_array = [state[0]; DIGEST_SIZE];
        apply_permutation(&mut state);
        digest_array[1] = state[0];

        AnemoiDigest::new(digest_array)
    }
}

impl Jive<Felt> for AnemoiHash {
    /// This instantiation of the Anemoi hash doesn't support the
    /// Jive compression mode due to its targeted security level,
    /// incompatible with the output length of Jive.
    ///
    /// Implementers aiming at using this instantiation in a setting
    /// where a compression mode would be required would need to
    /// implement it directly from the sponge construction provided.
    fn compress(_elems: &[Felt]) -> Vec<Felt> {
        unimplemented!()
    }

    /// This instantiation of the Anemoi hash doesn't support the
    /// Jive compression mode due to its targeted security level,
    /// incompatible with the output length of Jive.
    ///
    /// Implementers aiming at using this instantiation in a setting
    /// where a compression mode would be required would need to
    /// implement it directly from the sponge construction provided.
    fn compress_k(_elems: &[Felt], _k: usize) -> Vec<Felt> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::super::BigInteger384;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![Felt::new(BigInteger384([
                0x74cc7bf163640c33,
                0x14608c048284f554,
                0x69781fa9d9462819,
                0xb223a6843b74cbd4,
                0x39628d3baffe19c5,
                0x008ce7d95a446491,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0xd96fccafb5f76b19,
                    0xc7d1cf4d715ae188,
                    0x6708326e71b9423e,
                    0xd39a50ea2d5b2d4e,
                    0x7efd89dea17ca6b8,
                    0x014b4a391e89bcac,
                ])),
                Felt::new(BigInteger384([
                    0x1cb60d113312f886,
                    0x0eca444c10a4d5be,
                    0x79771f2ef3d15383,
                    0xb68b8ae47ed40538,
                    0xf7b678880d4e0fa3,
                    0x016143dcc377f76c,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x4d2eaccc79664162,
                    0xfee8cedc0a23d156,
                    0x0f6869ec7c3c0ab5,
                    0x9b8ba28371ac1b3a,
                    0xbe700570946be768,
                    0x00a2ac05aa73caba,
                ])),
                Felt::new(BigInteger384([
                    0x4808b37e83c6523a,
                    0xfd8ba37b2d02505d,
                    0x5f1b9bab6fae4f82,
                    0x4a011b53c026460d,
                    0x956ebb5a4e566cc5,
                    0x000022d5d8303db3,
                ])),
                Felt::new(BigInteger384([
                    0x0662985467675304,
                    0xab0e50255d6a2cca,
                    0x2eb1e8f5d52625b3,
                    0x88759a07329f1548,
                    0xf7080247ad73324f,
                    0x00dfcad0f894f38d,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xcf5b7c4451efb6cf,
                    0xe131c53a67741cb2,
                    0xb4316458b4888434,
                    0xf5c3d30ad6941798,
                    0x8ec2a054a4be03ea,
                    0x0001341113775b0d,
                ])),
                Felt::new(BigInteger384([
                    0x7993d43eafbc0943,
                    0x803c265c6ae1e10c,
                    0x2e1c71d929ca91f3,
                    0xecfed5ebd449cf24,
                    0xaa1f9cffd3a28579,
                    0x01397c5e75325d42,
                ])),
                Felt::new(BigInteger384([
                    0xeadc8014b6be8a13,
                    0xea2a7abb41e7575f,
                    0x8e9a99261a7be39b,
                    0xdf13c394002ba00b,
                    0x271073cce3f8e381,
                    0x01832f1e709bc591,
                ])),
                Felt::new(BigInteger384([
                    0xb280ee91a5354272,
                    0x8a71243c306cf9d0,
                    0xc3c3e74b444056da,
                    0x04a631b922f2aecf,
                    0xf34e39c63d204b73,
                    0x00ac89be9a129d73,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x2a34d702687f8c1e,
                    0xb4cdcbd8607ff1c8,
                    0xfde28089efbc1acb,
                    0xda2449c2322ac82f,
                    0xf60911210b7475f9,
                    0x019207ef6cb5636a,
                ])),
                Felt::new(BigInteger384([
                    0xbdec2cbb9c26a24f,
                    0x2f84d98823ba5736,
                    0x0fdb4110cc33e7ea,
                    0xb3d062c14cbd2271,
                    0x77dc70a6988ad187,
                    0x015e24bd2c27c661,
                ])),
                Felt::new(BigInteger384([
                    0x995ed8045b9c3355,
                    0x036cbd2417eea868,
                    0xa259b3bc9d875a66,
                    0xc2c4baf02cdb5a2a,
                    0xc8069e462cb3df3c,
                    0x009ceb7e49b04bcf,
                ])),
                Felt::new(BigInteger384([
                    0x238843fa98b0bcc1,
                    0x3b2a319c969354ff,
                    0x3c01df4ea4735e64,
                    0xe1af72bd6540e0b9,
                    0x67b2fa170b63cca3,
                    0x017edaa5fc85ca9f,
                ])),
                Felt::new(BigInteger384([
                    0xcbbe464094d6a2f7,
                    0x10402396a26417fa,
                    0x66e6f786577691d4,
                    0xf7256bc3e2e331e2,
                    0x569302c9999b6a78,
                    0x0118baaee18ea97b,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x8c5dce671d5b3e09,
                    0xc7d02d70eec00f0f,
                    0x9b1a42fed96edc4e,
                    0x0ff85e4eeea3cf84,
                    0x7aec788d5cf9916c,
                    0x0079cfd99a5d750d,
                ])),
                Felt::new(BigInteger384([
                    0xc70865f51a4bcc8d,
                    0xb7438090afbab557,
                    0xd4eb51a84e55cfc0,
                    0xd5cc208a3bd79352,
                    0xa4423a43fa97fc41,
                    0x003f2c8b5e7c9d84,
                ])),
                Felt::new(BigInteger384([
                    0x007ca3041c5f6f46,
                    0xb331f8ad50a70d5c,
                    0x6d1ce0e8bd5aebbd,
                    0x322319b08795243a,
                    0x097de2e030bfd690,
                    0x0149516049254fee,
                ])),
                Felt::new(BigInteger384([
                    0x89fe3abe7f63b9fd,
                    0x72ee1026255638bf,
                    0x5a897b5c4f5dc2d9,
                    0x0c81753bc633ebb7,
                    0xd02bdf8bbe48eb0c,
                    0x0174787401f11a2e,
                ])),
                Felt::new(BigInteger384([
                    0xe9de9ac19e92673c,
                    0xc7e6e28e7e550a9c,
                    0x7fa677084e612af1,
                    0x703eb973b46371f9,
                    0x04a2ba1f8e9f7b18,
                    0x00b84186a3bd6d9f,
                ])),
                Felt::new(BigInteger384([
                    0xefab7a6f81041d62,
                    0x3945287296b712f8,
                    0x033ef923112149be,
                    0x3230d0a3d338ef66,
                    0xc2e45e3327a62ada,
                    0x01709cb03db285cb,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger384([
                    0xb4dd4ea1e1a2bb99,
                    0xa28412ee2d7d1318,
                    0x0932be31d39854bd,
                    0xa83dbac449ba2ba9,
                    0x409a9cba7456acc1,
                    0x012a9911b562af45,
                ])),
                Felt::new(BigInteger384([
                    0x78639f4833d913ed,
                    0x1ecc08a11820d6b0,
                    0xd18b68ae85cc8ee0,
                    0xa94e9bf1a2fd116a,
                    0x48502522585f9ad3,
                    0x01028b6e2a9bef92,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x2ca6fdf4a42aecb3,
                    0xe34bb94af317b5f4,
                    0xf19f532d9e5e08d7,
                    0x58e9a10af194c37c,
                    0xb8b95825ad6374d1,
                    0x00e23a16ab6db0fc,
                ])),
                Felt::new(BigInteger384([
                    0xbe468b89b4b31bbf,
                    0x089c0c4861388555,
                    0x69eaf6010ecf3b40,
                    0xd7800b959e2b2567,
                    0x1942c71a392a085e,
                    0x0021482da93d40db,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xd1a68580835ee327,
                    0xe95fbef7f777c5b5,
                    0x17425358de2ef15a,
                    0x05b108a90bf5b071,
                    0x48995e4f59153e8d,
                    0x007f78292e647acf,
                ])),
                Felt::new(BigInteger384([
                    0x1888d7f3bb1f588c,
                    0xbd616a6c8a80f58b,
                    0x16089f32e1176a8c,
                    0x0b8287ed41250b48,
                    0xb3d5b1579befc7ce,
                    0x012a2f29393cf08b,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x2c30a999e9c01074,
                    0xecbddbdf9a2a4b0e,
                    0xda415bc1c3757752,
                    0x872deb512056c410,
                    0x54d3e1483ebf7a2c,
                    0x00ec2500f38ac721,
                ])),
                Felt::new(BigInteger384([
                    0x19d602394eda40ff,
                    0x5b21121140af40d2,
                    0x3187e9395d074d44,
                    0x32c658b9b653980c,
                    0xce6017c5ad723711,
                    0x00cec047d47108cd,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x3ffda5d113d56d18,
                    0x5a998fda0a008ecb,
                    0xec24ec81652834bf,
                    0x7a8d6dd13048b63a,
                    0xccde1f70d3945643,
                    0x00ed7fb0f591723d,
                ])),
                Felt::new(BigInteger384([
                    0x2fa105079826c8a2,
                    0x0ad91f64d7ceb277,
                    0x9032054c71c957e2,
                    0x1778148fa1e88094,
                    0x322adc96fd5b7590,
                    0x016d1ac63f3fd1d3,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xa48fe16259311217,
                    0xef646f98cc4130ab,
                    0xe76c747103fe4e82,
                    0x9edc4adde521873d,
                    0x5db69a073dfe9c69,
                    0x007f2183a433ec14,
                ])),
                Felt::new(BigInteger384([
                    0x86457a50bc0b051e,
                    0x4eaa3bf6fa0849d1,
                    0xbe3af368c04c2134,
                    0x942e5d55160c11f2,
                    0xf3c3254f95f9bdca,
                    0x014c1f0a1108e287,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xe7680b62f5c91cdf,
                    0x2dcb175d212d41b8,
                    0xa4f105056fa7a21a,
                    0xae86c44554f88249,
                    0xea56616f64420daf,
                    0x00660cfc5237e4c5,
                ])),
                Felt::new(BigInteger384([
                    0x489ed4ca1d35d29a,
                    0x4586c81f8e939227,
                    0x482ca8467973ce33,
                    0x6f341c50d605bb14,
                    0xe0bf7f7359cbb2dd,
                    0x013ea84ee945d1e8,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x103a38a8eeed02f2,
                    0xc2b7ae3df6185d16,
                    0xbc268f4f80a8ff41,
                    0x490cc97160c74669,
                    0x770ad32c7b15a4dd,
                    0x015daa820e1b3bc8,
                ])),
                Felt::new(BigInteger384([
                    0xb4f4f102dbeeb202,
                    0x27a937c3c665626a,
                    0x9690cf3e05c760ea,
                    0x6f9be08ef49f6fdd,
                    0xab91b5c15ade813c,
                    0x0175e78e8686051e,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xc525d75984d0a0fe,
                    0xde8438114aabf0bb,
                    0x9584e58dfc80f6a7,
                    0xa2358e792abec57c,
                    0x10bb73f7a6e28356,
                    0x00dc6d87d10405db,
                ])),
                Felt::new(BigInteger384([
                    0xa7dedf3a41fbf14f,
                    0x35f548347e383c88,
                    0xf923c0a0514dc50c,
                    0x46866fab8af7e450,
                    0x9ad4fd045ca7363b,
                    0x00d5f00cf6269ea2,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xb3140276f6f627a3,
                    0x5ac43e54d7d00277,
                    0x0e7b38152fccd3d8,
                    0x1b91833e9411c42f,
                    0x4bb06d9d7642947e,
                    0x015199a40c567ecf,
                ])),
                Felt::new(BigInteger384([
                    0xc27d7cdba283119f,
                    0xd3f86e0bdfd43872,
                    0x8fa536f0de4d45ff,
                    0x50a7a8cd5c46840e,
                    0x9bfc481fe6cba569,
                    0x00731baf5aa3d828,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }
}
