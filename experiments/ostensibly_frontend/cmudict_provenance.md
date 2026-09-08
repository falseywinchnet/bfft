# CMUdict evaluation anchor

The word-candidate experiment measured against the CMUSphinx CMUdict master
tree at commit `74790861f652b15e4ac49015a90074ad62a27690`.

- Source file: `cmudict.dict`
- SHA-256: `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22`
- Parsed pronunciations with 1--18 phones in the declared 39-phone inventory:
  `134834`

The dictionary is not silently embedded in the engine. Runners require an
explicit path, record that file's SHA-256 in every result, strip lexical stress
digits for the current acoustic inventory, and preserve alternate
pronunciations and homophone word sets. CMUdict is maintained by Carnegie
Mellon University's Speech Group; retain its `LICENSE` when redistributing the
dictionary itself.
