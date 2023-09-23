//! Implementation of the Anemoi permutation

use super::{sbox, Felt, MontFp};
use crate::{Anemoi, Jive, Sponge};
use ark_ff::{One, Zero};
/// Digest for Anemoi
mod digest;
/// Sponge for Anemoi
mod hasher;
/// Round constants for Anemoi
mod round_constants;

pub use digest::AnemoiDigest;

// ANEMOI CONSTANTS
// ================================================================================================

/// Function state is set to 2 field elements or 64 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 2;
/// 1 element of the state is reserved for rate.
pub const RATE_WIDTH: usize = 1;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 1;

/// One element (32-bytes) is returned as digest.
pub const DIGEST_SIZE: usize = RATE_WIDTH;

/// The number of rounds is set to 21 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 21;

// ANEMOI INSTANTIATION
// ================================================================================================

/// An Anemoi instantiation over Jubjub basefield with 1 column and rate 1.
#[derive(Debug, Clone)]
pub struct AnemoiJubjub_2_1;

impl<'a> Anemoi<'a, Felt> for AnemoiJubjub_2_1 {
    const NUM_COLUMNS: usize = NUM_COLUMNS;
    const NUM_ROUNDS: usize = NUM_HASH_ROUNDS;

    const WIDTH: usize = STATE_WIDTH;
    const RATE: usize = RATE_WIDTH;
    const OUTPUT_SIZE: usize = DIGEST_SIZE;

    const ARK_C: &'a [Felt] = &round_constants::C;
    const ARK_D: &'a [Felt] = &round_constants::D;

    const GROUP_GENERATOR: u32 = sbox::BETA;

    const ALPHA: u32 = sbox::ALPHA;
    const INV_ALPHA: Felt = sbox::INV_ALPHA;
    const BETA: u32 = sbox::BETA;
    const DELTA: Felt = sbox::DELTA;

    fn exp_by_inv_alpha(x: Felt) -> Felt {
        sbox::exp_by_inv_alpha(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sbox() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let mut input = [
            [Felt::zero(), Felt::zero()],
            [Felt::one(), Felt::one()],
            [Felt::zero(), Felt::one()],
            [Felt::one(), Felt::zero()],
            [
                MontFp!(
                    "51687309537499687832787312941976599784064532977316092918533053386538748425642"
                ),
                MontFp!(
                    "27245180385515469316683280807646167491575980832157009146585007068470321964262"
                ),
            ],
            [
                MontFp!(
                    "52175830888449177420799525156647727456986450309407845803758865897364847179738"
                ),
                MontFp!(
                    "39622503581685635500700621337279261352946169581417185398259207807755733091294"
                ),
            ],
            [
                MontFp!(
                    "19470931668995937460604715897699377052859387353563090962192068868608557288654"
                ),
                MontFp!(
                    "52428813015582117987474147227661578665291330159066236929548107648208211253768"
                ),
            ],
            [
                MontFp!(
                    "34725136430734279097783693363635397852820298036356042278369044959187154401979"
                ),
                MontFp!(
                    "9155944415537595472308318851850913269911307292288152885528758693209192915378"
                ),
            ],
            [
                MontFp!(
                    "15324022610495839874774224308496671270390212773715628021233240921036882048186"
                ),
                MontFp!(
                    "33157559832878686443229823975723354165092302139756182295381510740322578094918"
                ),
            ],
            [
                MontFp!(
                    "24325650607472239761644712985779158601176705943089386019707973129364398463560"
                ),
                MontFp!(
                    "36476596879636242194858777999278567147480546106023627833691137236169581850101"
                ),
            ],
        ];

        let output = [
            [
                MontFp!(
                    "14981678621464625851270783002338847382197300714436467949315331057125308909861"
                ),
                Felt::zero(),
            ],
            [
                MontFp!(
                    "26465412774926089111718321100574595527540964126946526979500696246159200663557"
                ),
                MontFp!(
                    "39248768654128688292658764022179785046913739642751806076985032101719802651189"
                ),
            ],
            [
                MontFp!(
                    "12740277426642017153170591585468114140384764416905683044706614677794597857620"
                ),
                MontFp!(
                    "14891594025153715433052427327081456214862500696375893038292967289916437753110"
                ),
            ],
            [
                MontFp!(
                    "14981678621464625851270783002338847382197300714436467949315331057125308909869"
                ),
                MontFp!(
                    "52435875175126190479447740508185965837690552500527637822603658699938581184512"
                ),
            ],
            [
                MontFp!(
                    "30539342768044772785240858832987438746638686553106686724949418598999958516685"
                ),
                MontFp!(
                    "20906572893529218028458150808451715646423977802151887706433959987398846705839"
                ),
            ],
            [
                MontFp!(
                    "22346066152187674141760538567730885692050154791927920531163343236293067435424"
                ),
                MontFp!(
                    "1456757276424856397682120883987169172441004806229947861007954668371527553243"
                ),
            ],
            [
                MontFp!(
                    "51263975795335663516585515610761004054567796038636487381764356241385176767693"
                ),
                MontFp!(
                    "47859076494749252787425002672564777642089326669133167948819513468280427613779"
                ),
            ],
            [
                MontFp!(
                    "36157767113889104730581830280171322656039874429546786630559027084275883062577"
                ),
                MontFp!(
                    "20664812872910031392220671031275616107468626814133329099881146874021905253695"
                ),
            ],
            [
                MontFp!(
                    "44059093636116542838780832359223492731500879977663313687014137779330288888147"
                ),
                MontFp!(
                    "9385016998954961867521296928435557987297037770801602747094261251148525113071"
                ),
            ],
            [
                MontFp!(
                    "8506406649966831848758866184864510991904659426118219847758868610647732969620"
                ),
                MontFp!(
                    "23128801098188204035035120131318699886048855401612657028783560188383728323528"
                ),
            ],
        ];

        for i in input.iter_mut() {
            AnemoiJubjub_2_1::sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
