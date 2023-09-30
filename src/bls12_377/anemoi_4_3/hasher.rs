//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiBls12_377_4_3, Jive, Sponge};
use super::{DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
use crate::traits::Anemoi;
use ark_ff::PrimeField;

use super::Felt;
use super::{One, Zero};

impl Sponge<Felt> for AnemoiBls12_377_4_3 {
    type Digest = AnemoiDigest;

    fn hash(bytes: &[u8]) -> Self::Digest {
        // Compute the number of field elements required to represent this
        // sequence of bytes.
        let num_elements = if bytes.len() % 47 == 0 {
            bytes.len() / 47
        } else {
            bytes.len() / 47 + 1
        };

        let sigma = if num_elements % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 47-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut i = 0;
        let mut num_hashed = 0;
        let mut buf = [0u8; 48];
        for chunk in bytes.chunks(47) {
            if num_hashed + i < num_elements - 1 {
                buf[..47].copy_from_slice(chunk);
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
            state[i] += Felt::from_le_bytes_mod_order(&buf[..]);
            i += 1;
            if i % RATE_WIDTH == 0 {
                AnemoiBls12_377_4_3::permutation(&mut state);
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
        // to the whole state.
        if sigma.is_zero() {
            state[i] += Felt::one();
            AnemoiBls12_377_4_3::permutation(&mut state);
        }

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
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
                AnemoiBls12_377_4_3::permutation(&mut state);
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
            AnemoiBls12_377_4_3::permutation(&mut state);
        }

        // Squeezing phase

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn merge(digests: &[Self::Digest; 2]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        // 2*DIGEST_SIZE < RATE_SIZE so we can safely store
        // the digests into the rate registers at once
        state[0..DIGEST_SIZE].copy_from_slice(digests[0].as_elements());
        state[DIGEST_SIZE..2 * DIGEST_SIZE].copy_from_slice(digests[0].as_elements());

        // Apply internal Anemoi permutation
        AnemoiBls12_377_4_3::permutation(&mut state);

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiBls12_377_4_3 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.to_vec();
        AnemoiBls12_377_4_3::permutation(&mut state);

        let mut result = [Felt::zero(); NUM_COLUMNS];
        for (i, r) in result.iter_mut().enumerate() {
            *r = elems[i] + elems[i + NUM_COLUMNS] + state[i] + state[i + NUM_COLUMNS];
        }

        result.to_vec()
    }

    fn compress_k(elems: &[Felt], k: usize) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);
        assert!(STATE_WIDTH % k == 0);
        assert!(k % 2 == 0);

        let mut state = elems.to_vec();
        AnemoiBls12_377_4_3::permutation(&mut state);

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
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::super::MontFp;
    use super::*;
    use ark_ff::BigInteger;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![MontFp!(
                "151784395159840881100476615820860125446208590563016900253994611973476120876020739211056415913863582716187750670325"
            )],
            vec![
                MontFp!(
                    "248332283272864683704036740560337592654887163711254580843329153286830464649325062580169894462323263526723015056868"
                ),
                MontFp!("73872971620594878519579519870883325381917650130994887007487546333757748944870382425707613478642638198928278622987"
                ),
            ],
            vec![
                MontFp!(
                    "253404415228193614695833183326000116355203070746791246285647009929594735741912917205000350665772868226095764483542"
                ),
                MontFp!("200808382639976305457513271520374825989196919411760545885343200642320153502354224951365756566911650724585511350853"
                ),
                MontFp!("214853645096049440262169487905844674327252125483157699607129285129242631790482015824934365478334441610585581765646"
                ),
            ],
            vec![
                MontFp!(
                    "24452898258044290105682032131010953719966577825934286587455692440936430712926423662923951180719449011781755849091"
                ),
                MontFp!("68036111236254983016464719935388750487151085306644857463718636878913915660717422899508805376164291132936305651549"
                ),
                MontFp!("119814581033229604224588304484000940913346715258310505533671062424278330336495891273358810620038961552132570344805"
                ),
                MontFp!("146786360464378992753351732073036341474646388118635387105069745454372944846535403582379785641449915081441685241831"
                ),
            ],
            vec![
                MontFp!(
                    "232934047713835826189642755067598105652402953051462186411057972145784801688247151504568243631412655037565837575757"
                ),
                MontFp!("51891050044658712786449936018589028012756027654536350357756365097298627561196049012242317876852037092416560910103"
                ),
                MontFp!("140157604933313092669364803748161937831197622223558982346762823892285695108052895968044812973568844051340421419650"
                ),
                MontFp!("49063998238900713944259696184560028117041571852708162986701374121301961792895189609933634465196413916200608999619"
                ),
                MontFp!("182715104911529551188014228555616416212700779904846405068176750516201576848584786270331419395711573173280742513022"
                ),
            ],
            vec![
                MontFp!(
                    "13904599908677035213543036116655815491171936969788883080056877419504094113088226632022301066280953350468512812569"
                ),
                MontFp!("122063089433734133607841584654605500874145833119319972501793931698322982189001473090573809475872718234572446338103"
                ),
                MontFp!("15686995896518365224322596649751290886569484747533689471723426828889426595002623585601349341253395261532893685407"
                ),
                MontFp!("237746465690590083885447534720888811244542841745670503453103593821753316374771484039116200688928220005868730618826"
                ),
                MontFp!("26332686776066989841849277684720820158149273328296092046334949193988732739237302738932314605293198099976982372239"
                ),
                MontFp!("154913739715427863803657124516740821701823267856775793724677864378720872186808431937529555862818757741509783020532"),],];

        let output_data = [
[MontFp!("164656778089067776870692351061562088286510627838214779111394227597199991499276054482462813901307983845389960630090"),],
[MontFp!("235421008423750222113281425197065350815205915036608889296658377566471443802789161599450202659856007754310347477866"),],
[MontFp!("52561647789853035454769336399420842077931476677317792325459253622064483797729294479663760385974644902213972852210"),],
[MontFp!("148331049896119772678387059133407070018742451702702105443281236836098980134097884656360262254427019930703627108449"),],[MontFp!("31888388626901995919721279273893574682432785481699663022004877447888597556400517903663614962930367609017392191097"),],[MontFp!("84580302935893226435006985163714257490725017853893641106247246408162293524807496656152024223821812466003251199220"),],[MontFp!("5299509691582890633117867007514018316624780870273985334153052614015108768512614711811521332843694174907839813735"),],[MontFp!("32831090461859823760523120645535374703192680620427237537195517670681347005107643592300385445787803975785139290939"),],[MontFp!("36262178947576995011848214793869236110629213608374695078641684231497817975109165522638670763942499871170488113379"),],[MontFp!("181176227309696350315188533762560236311761993511085674346185500750188294069797074658199825922899220617333168413469"),],];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiBls12_377_4_3::hash_field(input).to_elements()
            );
        }
    }

    #[test]
    fn test_anemoi_hash_bytes() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [MontFp!("164656778089067776870692351061562088286510627838214779111394227597199991499276054482462813901307983845389960630090"),],
[MontFp!("235421008423750222113281425197065350815205915036608889296658377566471443802789161599450202659856007754310347477866"),],
[MontFp!("52561647789853035454769336399420842077931476677317792325459253622064483797729294479663760385974644902213972852210"),],
[MontFp!("148331049896119772678387059133407070018742451702702105443281236836098980134097884656360262254427019930703627108449"),],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 188];
            bytes[0..47].copy_from_slice(&input[0].into_bigint().to_bytes_le()[0..47]);
            bytes[47..94].copy_from_slice(&input[1].into_bigint().to_bytes_le()[0..47]);
            bytes[94..141].copy_from_slice(&input[2].into_bigint().to_bytes_le()[0..47]);
            bytes[141..188].copy_from_slice(&input[3].into_bigint().to_bytes_le()[0..47]);

            assert_eq!(expected, AnemoiBls12_377_4_3::hash(&bytes).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
[MontFp!("168857760056935172846042246077605425854459529921437979876872555197644520166747086657350313124820327939581939546688"
                ),
                MontFp!("227578620726741137788011841074791103971961456654996462159216243608183452768222827984447900301510222033867890347639"),],[MontFp!("33878933403548795575101163153844627990806945952461898033595620094787622876712560160782042310211263041687029151988"
                ),
                MontFp!("226214177850122047293279773806027076328962453844200429102468221766085032652542508353736465840229800774462000982421"),],[MontFp!("146695613033298421833912302030457254120471013186575211803075449856875630402917014212884714513418290448707449862982"
                ),
                MontFp!("91202527302881221223190834557990245347278629206286693432478459534162484484892425697592309248613383397717123420648"),],[MontFp!("154580544948945795763395643897120619788435069582860116340372530979178418435899924057750580686784910258567018528930"
                ),
                MontFp!("43241772285527117250498190204128144589162182818500867264472744715727478714926005616292529722845675633183874884263"),],];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBls12_377_4_3::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBls12_377_4_3::compress_k(input, 2));
        }

        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [MontFp!("137771954770707216623401353457502996290027473821519781496204536139107504586629091866829325286757189849009508436150"),],[MontFp!("1428685240701748857728203264978170783375887041747666596179579194152187180914245739549620010867703691708708676232"),],[MontFp!("237898140336179643057103136588447499467749642392861905235553909391038114887809439910477023762031673846424573283630"),],[MontFp!("197822317234472913013893834101248764377597252401360983604845275694905897150825929674043110409630585891750893413193"),],];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBls12_377_4_3::compress_k(input, 4));
        }
    }
}
