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

And then, in your program:

```rust
extern crate markovr;

pub fn main() {
    // Create a new, first-order Markov Chain.
    let mut m = markovr::MarkovChain::new(1);

    // alpha will be both our encoding mapping and training data.
    // markovr only speaks u64s, so the indices of alpha will be the encoding.
    let alpha: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();

    // Train the model.
    for i in 1..alpha.len() {
        m.train(&[(i - 1) as u64], i as u64, 1);
    }

    // Generate values from the model.
    let mut last: Option<u64> = Some(0);
    while last.is_some() {
        print!("{} ", alpha[last.unwrap() as usize]);
        last = m.generate(&[last.unwrap()]);
    }
    // Prints: a b c d e f g h i j k l m n o p q r s t u v w x y z

    // What's the probability that 'z' follows 'y'?
    print!("\n{}", m.probability(&[24], 25));
    // Prints: 1
    // What's the probability that 'z' follows 'a'?
    print!("\n{}\n", m.probability(&[0], 25));
    // Prints: 0
}
```

It's important to note that markovr only handles u64s at the moment.


## Future Plans

Support more types than u64 so mapping is not necessary.
