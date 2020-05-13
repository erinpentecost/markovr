mod die;
use cfg_if::cfg_if;
use std::collections::HashMap;
use std::convert::TryFrom;

pub trait Element: Eq + PartialEq + Copy + Clone + std::hash::Hash {}
impl<T> Element for T where T: Eq + PartialEq + Copy + Clone + std::hash::Hash {}

/// Variable-order Markov chain.
pub struct MarkovChain<T: Element> {
    // the 'memory' for the MarkovChain chain.
    // 1 is the typical MarkovChain chain that only looks
    // back 1 element.
    // 0 is functionally equivalent to a weighteddie.
    order: usize,
    // the number of elements in the key should
    // exactly equal the order of the MarkovChain chain.
    // missing elements should be represented as None.
    probability_map: HashMap<Vec<Option<T>>, die::WeightedDie<T>>,
    optional_elements: Vec<usize>,
}

impl<T: Element> MarkovChain<T> {
    /// Creates a new MarkovChain.
    ///
    /// 'order' is the order of the Markov Chain.
    ///
    /// A value of 1 is your typical Markov Chain,
    /// that only looks back one place.
    ///
    /// A value of 0 is just a weighted die, since
    /// there is no memory.
    ///
    /// You could think of order as the shape of the
    /// input tensor, where the input tensor is the
    /// sliding view / window.
    ///
    /// 'optional_keys' allows you to specify None for
    /// the elements in the given indices during
    /// generation. This drastically increases the memory
    /// usage by 2^optional_elements.len().
    pub fn new(order: usize, optional_elements: &[usize]) -> Self {
        // filter out optional elements that are too big.
        let opts: Vec<usize> = optional_elements
            .clone()
            .into_iter()
            .map(|i| *i)
            .filter(|i| *i < order)
            .collect();
        MarkovChain {
            order,
            probability_map: HashMap::<Vec<Option<T>>, die::WeightedDie<T>>::new(),
            optional_elements: opts,
        }
    }

    /// Truncates elements as needed
    fn to_partial_key(order: usize, view: &[Option<T>]) -> Vec<Option<T>> {
        view.into_iter()
            .skip(view.len() - order)
            .take(order)
            .cloned()
            .collect()
    }

    /// Truncates elements as needed
    fn to_full_key(order: usize, view: &[T]) -> Vec<Option<T>> {
        view.into_iter()
            .skip(view.len() - order)
            .take(order)
            .cloned()
            .map(|e| Some(e))
            .collect()
    }

    fn permute(
        key: Vec<Option<T>>,
        optionals: Vec<usize>,
        mut perms: Vec<Vec<Option<T>>>,
    ) -> Vec<Vec<Option<T>>> {
        if optionals.len() == 0 {
            perms.push(key);
            perms
        } else {
            let mut off = key.clone();
            off[optionals[0]] = None;
            let on = key.clone();

            perms = Self::permute(off, optionals[1..].to_vec(), perms);
            perms = Self::permute(on, optionals[1..].to_vec(), perms);
            perms
        }
    }

    // this generates 2^(number of optional keys) keys.
    // this is used during training.
    fn permute_key(&mut self, key: Vec<T>) -> Vec<Vec<Option<T>>> {
        let optioned_key: Vec<Option<T>> = key.into_iter().map(|e| Some(e)).collect();
        Self::permute(optioned_key, self.optional_elements.clone(), vec![])
    }

    /// Feeds training data into the model.
    ///
    /// 'view' is the sliding window of elements to load
    /// into the MarkovChain chain. the number of elements in
    /// view should be self.order + 1 (excess will be ignored).
    /// the last element
    /// in view is the element to increase the weight of.
    ///
    /// 'weight_delta' should be the number of times we're
    /// loading this view into the model (typically 1 at
    /// a time).
    pub fn train(&mut self, view: &[T], result: T, weight_delta: i32) {
        for partial_key in self.permute_key(view.clone().to_vec()) {
            // Train not just on the full key, but all partial ones as well.
            self.probability_map
                .entry(partial_key)
                .and_modify(|d| {
                    d.modify(result, weight_delta);
                })
                .or_insert((|| match u32::try_from(weight_delta).ok() {
                    Some(v) => die::WeightedDie::new(vec![die::WeightedSide {
                        element: result,
                        weight: v,
                    }]),
                    None => die::WeightedDie::new(vec![]),
                })());
        }
    }

    /// Generates the next value, given the previous item(s).
    ///
    /// view is the sliding window of the latest elements.
    /// only the last self.order elements are looked at.
    ///
    /// rand_val allows for a deterministic result, if supplied.
    pub fn generate_deterministic_from_partial(
        &self,
        view: &[Option<T>],
        rand_val: u64,
    ) -> Option<T> {
        let key = MarkovChain::to_partial_key(self.order, view);

        match self.probability_map.get(&key) {
            Some(v) => v.roll(Some(rand_val)),
            None => None,
        }
    }

    /// Generates the next value, given the previous item(s).
    ///
    /// view is the sliding window of the latest elements.
    /// only the last self.order elements are looked at.
    ///
    /// rand_val allows for a deterministic result, if supplied.
    pub fn generate_deterministic(&self, view: &[T], rand_val: u64) -> Option<T> {
        let key = MarkovChain::to_full_key(self.order, view);

        match self.probability_map.get(&key) {
            Some(v) => v.roll(Some(rand_val)),
            None => None,
        }
    }

    cfg_if! {
        if #[cfg(feature = "rand")] {
            /// Generates the next value, given the previous item(s).
            ///
            /// view is the sliding window of the latest elements.
            /// only the last self.order elements are looked at.
            pub fn generate(&self, view: &[T]) -> Option<T> {
                let key = MarkovChain::to_full_key(self.order, view);

                match self.probability_map.get(&key) {
                    Some(v) => v.roll(None),
                    None => None,
                }
            }

            /// Generates the next value, given the previous item(s).
            ///
            /// view is the sliding window of the latest elements.
            /// only the last self.order elements are looked at.
            pub fn generate_from_partial(&self, view: &[Option<T>]) -> Option<T> {
                let key = MarkovChain::to_partial_key(self.order, view);

                match self.probability_map.get(&key) {
                    Some(v) => v.roll(None),
                    None => None,
                }
            }
        }
    }

    /// Returns the probability of getting 'result', given
    /// 'view'.
    pub fn probability(&self, view: &[Option<T>], result: T) -> f32 {
        let key = MarkovChain::to_partial_key(self.order, view);

        let map = self.probability_map.get_key_value(&key);
        match map {
            Some(v) => v.1.get_probability(result),
            None => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let m0 = MarkovChain::new(0, &[]);
        assert_eq!(m0.generate(&[]), None);
        assert_eq!(m0.generate(&[1]), None);
        assert_eq!(m0.generate_deterministic(&[], 33), None);

        let m1 = MarkovChain::new(1, &[]);
        assert_eq!(m1.generate(&[1]), None);
        assert_eq!(m1.generate(&[1, 1]), None);
        assert_eq!(m1.generate_deterministic(&[1], 33), None);

        let m2 = MarkovChain::new(2, &[]);
        assert_eq!(m2.generate(&[1, 1]), None);
        assert_eq!(m2.generate(&[1, 1, 1]), None);
        assert_eq!(m2.generate_deterministic(&[1, 1], 33), None);
    }

    #[test]
    fn alphabet_first_order() {
        let mut m = MarkovChain::new(1, &[]);

        // this could have just been a number range,
        // but it serves as an example of how to encode
        // an alphabet
        let alpha: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
        let encoded: Vec<u64> = alpha
            .clone()
            .into_iter()
            .enumerate()
            .map(|(i, _x)| i as u64)
            .collect();

        for i in m.order..encoded.len() {
            m.train(&[encoded[i - 1]], encoded[i], 1);
        }

        for i in 0..(encoded.len() - 1) {
            let next = m.generate(&[encoded[i]].clone());
            match next {
                Some(v) => assert_eq!(v, encoded[i + 1]),
                None => panic!(
                    "can't predict next letter after {} (encoded as {})",
                    alpha[i], encoded[i]
                ),
            };
        }
    }

    #[test]
    fn alphabet_second_order() {
        let mut m = MarkovChain::new(2, &[]);

        // this could have just been a number range,
        // but it serves as an example of how to encode
        // an alphabet
        let alpha: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
        let encoded: Vec<u64> = (0..alpha.len()).map(|x| x as u64).collect();

        for i in m.order..encoded.len() {
            m.train(&[encoded[i - 2], encoded[i - 1]], encoded[i], 1);
        }

        for i in 1..(encoded.len() - 1) {
            let next = m.generate(&[encoded[i - 1], encoded[i].clone()]);
            match next {
                Some(v) => assert_eq!(v, encoded[i + 1]),
                None => panic!(
                    "can't predict next letter after {} (encoded as {})",
                    alpha[i], encoded[i]
                ),
            };
        }
    }
}
