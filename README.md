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
apt-get update -y
apt-get install espeak -y
```

## Notes

### Get syllables

```
import pronouncing
word = "hello"
syllables = pronouncing.phones_for_word(word)
len(syllables)
```