use cfg_if::cfg_if;
use std::convert::TryFrom;

#[cfg(feature = "rand")]
use rand::Rng;

#[cfg(feature = "serializer")]
use serde::{Deserialize, Serialize};

use super::Element;

/// This is a weighted die. You can add sides (faces),
/// change their weights, and so on.
#[derive(Clone, Eq, PartialEq)]
#[cfg_attr(feature = "serializer", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serializer",
    serde(bound = "T: Serialize, for<'t> T: Deserialize<'t>")
)]
pub struct WeightedDie<T: Element> {
    /// An element and its probabalistic weight,
    /// compared with its peers.
    items: Vec<T>,

    /// Caching running weights in order to support
    /// O(lg n) rolls.
    running_weight: Vec<u64>,
}

impl<T: Element> std::fmt::Debug for WeightedDie<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Element")
    }
}

impl<T: Element> WeightedDie<T> {
    /// Create a new weighted die.
    pub fn new() -> Self {
        WeightedDie::<T> {
            items: vec![],
            running_weight: vec![],
        }
    }

    fn find_first(&self, element: T) -> Option<usize> {
        let found_val = self
            .items
            .iter()
            .enumerate()
            .filter(|v| *v.1 == element)
            .next()
            .map(|v| v.0);
        found_val
    }

    fn gcd(a: u64, b: u64) -> u64 {
        let mut a = a;
        let mut b = b;
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    fn less_lossy_divide(numerator: u64, denominator: u64) -> f32 {
        let mut n = numerator;
        let mut d = denominator;
        let gcd = Self::gcd(n, d);
        n = n / gcd;
        d = d / gcd;
        return n as f32 / d as f32;
    }

    /// Returns the probability of rolling the selected side.
    /// If the die does not contain the side, returns 0.
    /// This will be off since floats aren't exact sometimes.
    pub fn get_probability(&self, element: T) -> f32 {
        match self.find_first(element) {
            Some(v) => {
                // if there is some found element, then
                // running_weight is not empty.
                Self::less_lossy_divide(
                    self.get_item_weight(v),
                    *self.running_weight.last().unwrap_or(&1),
                )
            }
            None => 0.0,
        }
    }

    fn get_item_weight(&self, idx: usize) -> u64 {
        match idx {
            0 => self.running_weight[idx],
            _ => {
                let prev_weight = self.running_weight.get(idx - 1).unwrap_or(&0);
                self.running_weight[idx] - prev_weight
            }
        }
    }

    fn modify_weight_by_idx(&mut self, idx: usize, weight_delta: i32) {
        let abs_delta = u64::try_from(weight_delta.abs()).ok().unwrap_or(0);
        if weight_delta > 0 {
            // the delta is positive. simple case.
            for i in idx..self.running_weight.len() {
                self.running_weight[i] += abs_delta;
            }
        } else {
            // need to reduce weight for some reason.
            let cur_weight = self.get_item_weight(idx);
            if abs_delta >= cur_weight {
                // need to remove or set to 0 weight
                for i in idx..self.running_weight.len() {
                    self.running_weight[i] -= cur_weight;
                }
            } else {
                // don't remove
                for i in idx..self.running_weight.len() {
                    self.running_weight[i] -= abs_delta;
                }
            }
        }
    }

    /// Modifies the weight of an element in the collection.
    /// If it doesn't exist, will add to the collection.
    /// Runs in O(n).
    pub fn modify(&mut self, elem: T, weight_delta: i32) {
        let found = self.find_first(elem);
        match found {
            Some(v) => {
                // In the collection, so modify it.
                self.modify_weight_by_idx(v, weight_delta);
            }
            None => {
                // Not in the collection, so add it.
                if weight_delta > 0 {
                    self.items.push(elem);
                    let preceding_weight = *self.running_weight.last().unwrap_or(&0);
                    self.running_weight.push(preceding_weight);
                    self.modify_weight_by_idx(self.running_weight.len() - 1, weight_delta);
                } else {
                    // nothing to do at all
                }
            }
        }
    }

    /// Select some element from the collection.
    /// This doesn't remove the element.
    /// roll is an optional param when you don't want
    /// to rely on a random value.
    /// Runs in O(lg n).
    pub fn roll(&self, roll: Option<u64>) -> Option<T> {
        let total_weight = *self.running_weight.last().unwrap_or(&0);

        // If there is nothing to roll, return nothing.
        if self.items.len() == 0 || total_weight == 0 {
            return None;
        }

        // Figure out the roll value, if supplied.
        let roll_result: u64 = match roll {
            Some(r) => r % total_weight,
            None => {
                cfg_if! {
                    if #[cfg(feature = "rand")] {
                        let mut rng = rand::thread_rng();
                        rng.gen_range(0, total_weight) as u64
                    } else {
                        panic!("'roll' param is not optional when the 'rand' feature is off.");
                    }
                }
            }
        };

        // Binary search for the matching element.
        let mut start: usize = 0;
        let mut end: usize = self.items.len() - 1;
        while start <= end {
            let mid = (end + start) / 2;
            let matched = self.running_weight[mid];

            let mut one_less: u64 = 0;
            if mid > 0 {
                one_less = self.running_weight[mid - 1];
            }

            if matched > roll_result {
                if one_less <= roll_result {
                    // lt current element, but gte than
                    // the next smallest = we got our match.
                    return Some(self.items[mid]);
                } else {
                    // further to the left
                    end = mid - 1;
                }
            } else {
                // further to the right
                start = mid + 1;
            }
        }

        return Some(self.items[start]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(n: u64) -> Option<u64> {
        Some(n)
    }

    #[test]
    fn empty() {
        let c: WeightedDie<u64> = WeightedDie::new();
        assert_eq!(c.roll(Some(0)), None);
    }

    #[test]
    fn one_from_insertion() {
        let mut c = WeightedDie::new();
        c.modify(99, 3);
        assert_eq!(c.roll(Some(0)), w(99));
        assert_eq!(c.roll(Some(1)), w(99));
        assert_eq!(c.roll(Some(2)), w(99));
    }

    #[test]
    fn coin() {
        let mut c = WeightedDie::new();
        c.modify(1, 1);
        c.modify(2, 1);

        assert_eq!(c.items.len(), 2);

        assert_eq!(c.roll(Some(0)), w(1));
        assert_eq!(c.roll(Some(1)), w(2));
        assert_eq!(c.roll(Some(2)), w(1)); // rolled over
    }

    #[test]
    fn six_sided_die() {
        let mut c = WeightedDie::new();
        c.modify(1, 1);
        c.modify(2, 1);
        c.modify(3, 1);
        c.modify(4, 1);
        c.modify(5, 1);
        c.modify(6, 1);

        assert_eq!(c.items.len(), 6);

        assert_eq!(c.roll(Some(0)), w(1));
        assert_eq!(c.roll(Some(1)), w(2));
        assert_eq!(c.roll(Some(2)), w(3));
        assert_eq!(c.roll(Some(3)), w(4));
        assert_eq!(c.roll(Some(4)), w(5));
        assert_eq!(c.roll(Some(5)), w(6));
        assert_eq!(c.roll(Some(6)), w(1)); // rolled over
    }

    #[test]
    fn coin_with_edge() {
        let mut c = WeightedDie::new();
        c.modify(1, 100);
        c.modify(2, 1);
        c.modify(3, 100);

        assert_eq!(c.items.len(), 3);

        assert_eq!(c.roll(Some(0)), w(1));
        assert_eq!(c.roll(Some(98)), w(1));
        assert_eq!(c.roll(Some(99)), w(1));
        assert_eq!(c.roll(Some(100)), w(2));
        assert_eq!(c.roll(Some(101)), w(3));
        assert_eq!(c.roll(Some(200)), w(3));
        assert_eq!(c.roll(Some(230)), w(1)); // rolled over
    }

    #[test]
    fn modified_weight() {
        let mut c = WeightedDie::new();
        c.modify(1, 10);
        c.modify(2, 10);
        c.modify(1, 10);
        c.modify(3, 10);
        c.modify(2, -5);
        c.modify(3, -10);
        // at this point, we have (1, 20) and (2, 5).

        assert_eq!(c.roll(Some(0)), w(1));
        assert_eq!(c.roll(Some(9)), w(1));
        assert_eq!(c.roll(Some(19)), w(1));
        assert_eq!(c.roll(Some(20)), w(2));
        assert_eq!(c.roll(Some(29)), w(1)); // rolled over
    }

    #[test]
    fn get_prob() {
        let mut c = WeightedDie::new();
        c.modify(1, 10);
        c.modify(2, 10);
        c.modify(1, 10);
        c.modify(3, 10);
        c.modify(2, -5);
        c.modify(3, -10);
        // at this point, we have (1, 20) and (2, 5).

        let tmp = c.get_probability(1);
        assert_eq!(tmp, 20.0 / 25.0);
        assert_eq!(c.get_probability(2), 5.0 / 25.0);
        assert_eq!(c.get_probability(9), 0.0);
    }
}
