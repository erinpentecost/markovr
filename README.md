# markovr [![crates.io](https://img.shields.io/crates/v/markovr.svg)](https://crates.io/crates/markovr)

**Higher-order Markov Chains** can have longer memories than your [typical Markov Chain](https://en.wikipedia.org/wiki/Markov_chain), which looks back only 1 element. They are the basic building block for the [WaveFunctionCollapse](https://github.com/mxgmn/WaveFunctionCollapse) algorithm. A zeroth-order Markov Chain is the equivalent of a weighted die.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
markovr = {version = "0.2"}
```

Alternatively, if you don't want to bring in the [rand](https://crates.io/crates/rand) crate into your dependency tree:

```toml
[dependencies]
markovr = {version = "0.2", features = []}
```

Then it's as simple as this:

```rust
// Create a new, first-order Markov Chain.
let mut m = MarkovChain::new(1);

// Create a two-way mapping between your input data and u64s.
// Each of your inputs needs a unique u64 value.
// std::Hash can be your friend here, but let's do it ourselves.

// alpha will be both our encoding mapping and training data.
let alpha: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
// encoded is a parallel Vec to alpha that contains the u64 unique ids
// for each character.
let encoded: Vec<u64> = (0..alpha.len()).map(|x| x as u64).collect();

// Train the model.
for i in m.order..encoded.len() {
    m.train(&[encoded[i - 1]], encoded[i], 1);
}

// Generate values from the model.
for i in 0..(encoded.len() - 1) {
    let next = m.generate(&[encoded[i]].clone());
    // Since every input has exactly one output, the results
    // are deterministic.
    match next {
        Some(v) => assert_eq!(v, encoded[i + 1]),
        None => panic!(
            "can't predict next letter after {} (encoded as {})",
            alpha[i], encoded[i]
        ),
    };
}
```
