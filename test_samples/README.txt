TEST SAMPLES FOR THE AVIAN AI APP
=================================
All files come from the TEST sets of the new models, so the models never saw them during training.
"Expected" = what the trained model predicted in the Colab evaluation (your app should give the same
top-1 species; confidence can differ by a few percent because of float rounding / audio decoding).

IMAGES (image model v4, 200 species)
  01_Cardinal                 -> Cardinal                    (Colab confidence 96%)
  02_Blue_Jay                 -> Blue Jay                    (98%)
  03_Painted_Bunting          -> Painted Bunting             (93%)
  04_Mallard                  -> Mallard                     (97%)
  05_Ruby_throated_Hummingbird-> Ruby throated Hummingbird   (92%)
  06_Brown_Pelican            -> Brown Pelican               (98%)
  07_Pileated_Woodpecker      -> Pileated Woodpecker         (90%)
  08_American_Goldfinch       -> American Goldfinch          (92%)
  09_HARD_Common_Raven        -> WRONG on purpose: predicted American Crow (53%)  - raven/crow look almost identical
  10_HARD_Herring_Gull        -> WRONG on purpose: predicted Slaty backed Gull (60%) - young gulls are very hard

AUDIO (audio model v3, 70 species) -> files in test_samples\audio_v3\
  01_Northern_Cardinal        -> Northern Cardinal           (Colab confidence 81%)
  02_House_Sparrow            -> House Sparrow               (92%)
  03_Barn_Owl                 -> Barn Owl                    (96%)
  04_Mallard                  -> Mallard                     (92%)
  05_Canada_Goose             -> Canada Goose                (92%)
  06_Belted_Kingfisher        -> Belted Kingfisher           (94%)
  07_Red_Junglefowl           -> Red Junglefowl              (46%)
  08_Peregrine_Falcon         -> Peregrine Falcon            (90%)
  09_HARD_Rose-ringed_Parakeet -> WRONG on purpose: predicted House Sparrow (86%)
  Recordist and licence of every recording: test_samples\audio_v3\README.txt

  The old samples in test_samples\audio\ belong to audio model v2 (59 species) and are
  only used if AUDIO_MODEL_VERSION = "v2" in src\active_models.py.

Tip for the viva: show 2-3 correct ones, then one HARD one and explain why it is difficult.
