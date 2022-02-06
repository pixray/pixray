import pixray

pixray.run("pandas made of shiny metal")

pixray.run("pandas made of molten lava", outdir="outputs/fireout")

pixray.run("that's one content panda #pixelart", "pixel", outdir="outputs/pixel", )

pixray.run("an extremely hairy panda bear", "vdiff", custom_loss="aesthetic", outdir="outputs/hairout")

pixray.run("the ghost of a panda bear that died long ago", outdir="outputs/death", custom_loss="aesthetic")
