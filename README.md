# Dev setup

## Mac 


```
brew install espeak
```

```
pip3 install torch
```

## Linux

```
apt-get install espeak
```

## Notes

### Get syllables

```
import pronouncing
word = "hello"
syllables = pronouncing.phones_for_word(word)
len(syllables)
```