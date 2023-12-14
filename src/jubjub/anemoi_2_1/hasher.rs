//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiJubjub_2_1, Jive, Sponge};
use super::{DIGEST_SIZE, STATE_WIDTH};
use crate::traits::Anemoi;
use ark_ff::PrimeField;

use super::Felt;
use super::{One, Zero};

impl Sponge<Felt> for AnemoiJubjub_2_1 {
    type Digest = AnemoiDigest;

    fn hash(bytes: &[u8]) -> Self::Digest {
        // Compute the number of field elements required to represent this
        // sequence of bytes.
        let num_elements = if bytes.len() % 31 == 0 {
            bytes.len() / 31
        } else {
            bytes.len() / 31 + 1
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 31-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut buf = [0u8; 32];
        for (i, chunk) in bytes.chunks(31).enumerate() {
            if i < num_elements - 1 {
                buf[0..31].copy_from_slice(chunk);
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
            state[0] += Felt::from_le_bytes_mod_order(&buf[..]);
            AnemoiJubjub_2_1::permutation(&mut state);
        }
        state[STATE_WIDTH - 1] += Felt::one();

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn hash_field(elems: &[Felt]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        for &element in elems.iter() {
            state[0] += element;
            AnemoiJubjub_2_1::permutation(&mut state);
        }

        state[STATE_WIDTH - 1] += Felt::one();

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn merge(digests: &[Self::Digest; 2]) -> Self::Digest {
        // We use internally the Jive compression method, as compressing the digests
        // through the Sponge construction would require two internal permutation calls.
        let result = Self::compress(&Self::Digest::digests_to_elements(digests));
        Self::Digest::new(result.try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiJubjub_2_1 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.to_vec();
        AnemoiJubjub_2_1::permutation(&mut state);

        vec![state[0] + state[1] + elems[0] + elems[1]]
    }

    fn compress_k(elems: &[Felt], k: usize) -> Vec<Felt> {
        // This instantiation only supports Jive-2 compression mode.
        assert!(k == 2);

        Self::compress(elems)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use crate::jubjub::anemoi_2_1::AnemoiJubjub_2_1;

    use super::super::MontFp;
    use super::*;
    use ark_ff::BigInteger;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![MontFp!(
                "48175395693019743604712010240401407295263760496918609320567887350900617391938"
            )],
            vec![
                MontFp!(
                    "40454742959619504266124780247403051250769233161218450405944303623093451986416"
                ),
                MontFp!(
                    "8054087401217260540119647650149067888368265718118666773108585434053496512539"
                ),
            ],
            vec![
                MontFp!(
                    "8758999781725661589943067258778898714210579746607453110171065608650902955411"
                ),
                MontFp!(
                    "10178224565795089103313652432250635627412560984489920086835372540455062972043"
                ),
                MontFp!(
                    "29925420263172396940673207520643313079312157550992489658022590098553172519229"
                ),
            ],
            vec![
                MontFp!(
                    "43810900957132973642044031022126725744779901647233666513251283506650421253719"
                ),
                MontFp!(
                    "13869115914274770222885665907683059789190598851587796917340886888065950179345"
                ),
                MontFp!(
                    "19006550950025178457226798850946071923325417837656192809920080752737539580869"
                ),
                MontFp!(
                    "41347464737441972873801901743357452928767385147694625932616740287115542004184"
                ),
            ],
            vec![
                MontFp!(
                    "5641061656707918290766118699473598177131260713463794396205489094250322687059"
                ),
                MontFp!(
                    "1762472690591903627526805710995480828874871852953217753708762291919692776669"
                ),
                MontFp!(
                    "5442695738609165045860698972871486100783714976905853732598284193059726137484"
                ),
                MontFp!(
                    "44542361587146972628762599207869936445562095976920279660028208166325239459776"
                ),
                MontFp!(
                    "28063805515476868299441180197710909458474770498113403830730842276622910192214"
                ),
            ],
            vec![
                MontFp!(
                    "6222884466357505240696656069181876247031182115523656595681535984457165728442"
                ),
                MontFp!(
                    "24237625108810802359174427858350804364244467139304282092248149414791769682836"
                ),
                MontFp!(
                    "51505768400040663636491631745927718349620998271409928881692372400124132380182"
                ),
                MontFp!(
                    "25398536888248392215715464879479309238370329411183925527848925755051761566901"
                ),
                MontFp!(
                    "33121696194807230645952816752132160371820263776356020452728275724510567013833"
                ),
                MontFp!(
                    "35999566964686704391676648398286835098210554072067856490521627625893449004755"
                ),
            ],
        ];

        let output_data = [
            [MontFp!(
                "39323781703495850903956066402464124899786304471367493625790282776944096575660"
            )],
            [MontFp!(
                "4131620966514322712691579100546835054530941596511233645050286061964502598858"
            )],
            [MontFp!(
                "24324273559300271396953620022426323176594027249166939337399440665091996006791"
            )],
            [MontFp!(
                "42563550557900905328235558676622670923086275020825201990506124497842894441847"
            )],
            [MontFp!(
                "43692645056244253091360948912521218337025407173245382000695992868011997430113"
            )],
            [MontFp!(
                "48407239277773721892614795191886499809904530052733784927807524012932608104752"
            )],
            [MontFp!(
                "1548462962675086445132963037144064301548201331350500682572331921762552257306"
            )],
            [MontFp!(
                "19147993751018144465472900079428523048134768970599243883920905859942964907126"
            )],
            [MontFp!(
                "1031442954803170480786614934433496379556692105744131745792870583811175957592"
            )],
            [MontFp!(
                "41953808051964802716386912235165103363529406491121216029102398049652344033204"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiJubjub_2_1::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_hash_bytes() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
        ];

        let output_data = [
            [MontFp!(
                "39323781703495850903956066402464124899786304471367493625790282776944096575660"
            )],
            [MontFp!(
                "4131620966514322712691579100546835054530941596511233645050286061964502598858"
            )],
            [MontFp!(
                "24324273559300271396953620022426323176594027249166939337399440665091996006791"
            )],
            [MontFp!(
                "42563550557900905328235558676622670923086275020825201990506124497842894441847"
            )],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 62];
            bytes[0..31].copy_from_slice(&input[0].into_bigint().to_bytes_le()[0..31]);
            bytes[31..62].copy_from_slice(&input[1].into_bigint().to_bytes_le()[0..31]);

            assert_eq!(expected, AnemoiJubjub_2_1::hash(&bytes).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
        ];

        let output_data = [
            [MontFp!(
                "20387392009611881691526522206552322482509551426930619434849280967122120965518"
            )],
            [MontFp!(
                "44271465307610833975255894848424514129463332778307888463652844119174793376849"
            )],
            [MontFp!(
                "48081255911378449855577187903249438549252129278233190684459831383162273301317"
            )],
            [MontFp!(
                "20987090084610650936759092866102729962368129813397495848982616983114798496095"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiJubjub_2_1::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiJubjub_2_1::compress_k(input, 2));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiJubjub_2_1::merge(&[
                    AnemoiDigest::new([input[0]]),
                    AnemoiDigest::new([input[1]])
                ])
                .to_elements()
            );
        }
    }
}
