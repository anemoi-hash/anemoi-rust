//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiBn254_2_1, Jive, Sponge};
use super::{DIGEST_SIZE, STATE_WIDTH};
use crate::traits::Anemoi;
use ark_ff::PrimeField;

use super::Felt;
use super::{One, Zero};

impl Sponge<Felt> for AnemoiBn254_2_1 {
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
            AnemoiBn254_2_1::permutation(&mut state);
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
            AnemoiBn254_2_1::permutation(&mut state);
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

impl Jive<Felt> for AnemoiBn254_2_1 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.to_vec();
        AnemoiBn254_2_1::permutation(&mut state);

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
                "8794590961473411904012913256956674866225913057721735389120650157216849293309"
            )],
            vec![
                MontFp!(
                    "3145200566415442448691197067188750147559875286169943458059016307147140851337"
                ),
                MontFp!(
                    "6948511876435492649136441840414306836120842780275445630327690905592391419937"
                ),
            ],
            vec![
                MontFp!(
                    "7768269150808960338727016635529521023609446667578277771742273217198798857074"
                ),
                MontFp!(
                    "19031720944402394876154850067979927864606017002784129992021572353865733576806"
                ),
                MontFp!(
                    "21299305597107713470799051968075649019251798313971195547702159309262189815402"
                ),
            ],
            vec![
                MontFp!(
                    "8739794444373374437320561567398287787875152767337742498992255960974420564551"
                ),
                MontFp!(
                    "18055523130924247416289601815735394592750961886877711828231421773950074976142"
                ),
                MontFp!(
                    "2283560075286266658238072177861139695282914016559495468512314587710291549799"
                ),
                MontFp!(
                    "6516955896546720893507759968479438947681651087035799834256189139028417668849"
                ),
            ],
            vec![
                MontFp!(
                    "18868206257744691076171533540676365322752424511595293412714613296636836841428"
                ),
                MontFp!(
                    "13124043946363532942282918525840336221320875193003243864464947240562083555303"
                ),
                MontFp!(
                    "21675476711841002358533222763730202058673196295974588019497078882844178677171"
                ),
                MontFp!(
                    "21195164760374985994547294235549698827674740769600035700955236711701705122035"
                ),
                MontFp!(
                    "16826770706591427756364314617759023820357497662766823201950876360199989664303"
                ),
            ],
            vec![
                MontFp!(
                    "3228377566114929017106320912650322938507351180808295284245012197150523179963"
                ),
                MontFp!(
                    "16909065788345045925800817109077560065122579212259543033762951721067096311591"
                ),
                MontFp!(
                    "4825761689966844450630219774560546086184241805606275294446290334681343769054"
                ),
                MontFp!(
                    "21232877161693240578806429375775432669135047961960921397630845281136313272440"
                ),
                MontFp!(
                    "6942875049578575525713688488730704544006856910407705137130619086220864773064"
                ),
                MontFp!(
                    "11299219867659109485562968772566699969117450051804729254522958732965834335925"
                ),
            ],
        ];

        let output_data = [
            [MontFp!(
                "19343714080214199757292022909024009312436699556577457189881591835374641311754"
            )],
            [MontFp!(
                "8113497538289035899194181291131323688042826856528392025806280711737083916902"
            )],
            [MontFp!(
                "11909476273039149364186257796748244615541731997240663204105220020811083197092"
            )],
            [MontFp!(
                "21245387122202324495887734204687656469014050913569399481689643445634446612806"
            )],
            [MontFp!(
                "16808230860733105116680890524565357677174764156819276821359647806589203981684"
            )],
            [MontFp!(
                "19169356255747540911350285351685190904855147842053316673262974948275782775814"
            )],
            [MontFp!(
                "5905121874595833795618125889664078108057322454505380885810035358677829762734"
            )],
            [MontFp!(
                "21670483228380972108007068294704921256703456188837182403481546418320780192697"
            )],
            [MontFp!(
                "2621839791313312237147396950020428588046591993661449098136701343962174047268"
            )],
            [MontFp!(
                "4563543537134419823584603848847936302052234349785286643887775779581602681328"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiBn254_2_1::hash_field(input).to_elements());
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
                "19343714080214199757292022909024009312436699556577457189881591835374641311754"
            )],
            [MontFp!(
                "8113497538289035899194181291131323688042826856528392025806280711737083916902"
            )],
            [MontFp!(
                "11909476273039149364186257796748244615541731997240663204105220020811083197092"
            )],
            [MontFp!(
                "21245387122202324495887734204687656469014050913569399481689643445634446612806"
            )],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 62];
            bytes[0..31].copy_from_slice(&input[0].into_bigint().to_bytes_le()[0..31]);
            bytes[31..62].copy_from_slice(&input[1].into_bigint().to_bytes_le()[0..31]);

            assert_eq!(expected, AnemoiBn254_2_1::hash(&bytes).to_elements());
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
                "2317512736400239359977823018447801082849608015786769177609647583291382481920"
            )],
            [MontFp!(
                "21381856630176622904280416977152804235216222384802394731997901627399928695212"
            )],
            [MontFp!(
                "15966196448209272244335378655385867901323525781842763287895614621680333415655"
            )],
            [MontFp!(
                "5646550579889639872223774524937266947462036596705572174340703346042887657989"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBn254_2_1::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBn254_2_1::compress_k(input, 2));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiBn254_2_1::merge(&[
                    AnemoiDigest::new([input[0]]),
                    AnemoiDigest::new([input[1]])
                ])
                .to_elements()
            );
        }
    }
}
