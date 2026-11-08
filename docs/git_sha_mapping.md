# Git SHA mapping

The history from commit `3be56dd` onward was rewritten. Commit SHAs cited in run
records (for example `data/audit/test_access_log.jsonl`, frozen-run reports,
`data/audit/*` tables) and in the docs refer to the pre-rewrite history. Look them
up here to find the equivalent commit in the current history. Commits before
`3be56dd` are unchanged.

| Old SHA | New SHA | Subject |
|---|---|---|
| `3be56dda1d4ed9006354a81d03f09c5c74116bc7` | `eb6738902d944dbfc9b15d3d4b0e9778bee7a417` | Merge agent fleet runs, point-in-time evaluation and the LSTM comparison |
| `98359c8e723c33ba0d3f7a3d0418f599d66b0c99` | `93149a7c212c915ff65768da9811d0503ec4465b` | Fleet scoring: keep raw scores (calibration change not adopted) |
| `b2d64590b8635275f2add90ec0bbb2c0bb80ff19` | `f351b0c8643491ddd83f1b0619f41fb8be65005d` | Merged approach: survival and sequence models, ensembles, agent-fleet options |
| `a950e83f858401552af36c9c54e578cb3d50793c` | `9758c4e1726cdd31040b3e56598abeb85b22354a` | Make is_unbalance experiment variants actually use is_unbalance |
| `117542a7c2321643f016733b96288814d2fb2a15` | `fae0646287925dbd053a17a90b593dfa5d18c7a7` | Fail loudly when a two-stage run's second stage cannot be loaded |
| `39a05115c134ee4552f85c3d7a44d1819d5c2263` | `785ccf08964c585ff91b0e08bbe228a3c8cd2fa4` | docs: freeze the quarter-aligned run 839d92bc (runbook 1.1.1) |
| `6c04d5146f87b6e54a777567f41ef399c062c459` | `e2a1036b7a7a3377ebbdd2e44911a0054c0a5fab` | Add the frozen run 839d92bc bundle |
| `08397dc0369b67041102667209359103a9d73c22` | `f2b2fe2f04226afcbffef1e650851be7d3530e46` | train-merged: --tag output names, MLflow run, saved AFT/Cox boosters |
| `56b78d6d7e93648b06744e3d43d2b78167fb3783` | `9e38dbe5b9fb398613de4ec458982c1f1ddb6b89` | Add MERGED_TAG and refuse to overwrite existing run outputs |
| `25801da1714ecbad0a0d0c7c5e170bb79eabf169` | `be4b87f136a2d2b55b27eeb8d0a560292345af4b` | tests: send test-split access records to a temporary log |
| `56533e69a06bdc9280d64b1a2408f5d45718473f` | `662addf22619f323e463ba2259cb9653e3000015` | train: --tag so a run's outputs do not replace another's |
| `83a069ac8efc2e0be478e64101799f7a380f9efb` | `ab32ec980486f00672e10536597a0ce4d4ac7d8b` | Fix tagged merged scores in agent-fleet; tidy guard tests; document tags |
| `f50db7835a3160a99a995a8bb310491b23d9e3fe` | `80103117385d6a5b95be82e1068f155d1005eb67` | Merge train-merged run tags, MLflow logging and an overwrite guard |
| `e465403743cb36e0bea2092ce257b3c1a7b554a9` | `6ee3ce56317d67c25c717a9a6ddafb279dc0241a` | train-merged: check every score input before training anything |
| `d72bbd30546440efa5f227560c29e4a6ab8c11c5` | `79f3a74f938b4629bee21ee32e193317dd1c8a23` | Freeze the merged-approach run fb58d957 (runbook 1.1.2) |
| `451befd825d0f22862624f75de098b34310f4fb6` | `58a5df776a0d3b5714f6a03a88e930658fc5e6b0` | Pack the frozen run's score table for git; add pack/unpack make targets |
| `35b19607773fe16fb52e65911d406ec985368b43` | `009ec4544d60762eb449dde60fe52e12c2bf5f33` | Add make train-all: LightGBM, LSTM, TCN and train-merged under one tag |
| `f16b7ea31f01d0b44db769739a034e86ac2c0c4b` | `20ae9c8c3135151412de745c90f21f9390af45c2` | Merge the frozen merged-approach run fb58d957 and make train-all |
| `afd577733bc88ccc3d2c39ab0ff0832159f41083` | `205f3195b3f608bbb48beef3f831c51bd217dd72` | Agent loop per model of a train-merged run, and the result tables |
| `8af20a27c9d5ebe72ac16ee21ec46b43e86f6079` | `4766bed4d0f7333b52a2a62e912f71eee5c22d11` | Drop the (agent) suffix from the blend label in the results tables |
| `013cd6332905d9c687e50990ba6a01a3610bf6f0` | `98f22ff733734be2a314d405587f4847b7a85e72` | Add a validation-only search for the weights of a rank blend |
| `d48c83e3d9c06fef67066f6cf91d523dd4890df4` | `715df2dc73e032a674aef8fa1b297a731faff8d0` | Add a day-by-day replay of a recall-first alerting policy |
| `f4772f3c71d0202b1b3def06c4307463f037f9e1` | `af7ed04400898825ecfc738f6c53de09f0ca9036` | docs: record the current status of every model and blend (runbook 1.1.3) |
| `60e43bb4c950becd9001918ed21f1eb2cf60398b` | `f212a7829395e1e97b5e8f7ae462496333e7219d` | Compare wide Stage 1 nets for the two-stage cascade |
| `bfbe5908c2def26eb5ef2ea17a58909e123f99c3` | `c3daf0f305e02b841e5eec54a6a8b4462ad78701` | docs: add the wide-net cascade results to the status tables (runbook 1.1.3) |
| `8562ae19e3f6858bc2159f2c3d4aa4724e8f5dac` | `feb56b0a4fce21a78da2a58917b5b40185cd8599` | Generate the runbook 1.1.3 status tables from saved outputs |
| `f11beac5aa52a7a6af2e0a1bb83f0a44a1c2da52` | `2956047323f343b0199912a5b6768dbdb4d81e21` | Replace the merged agent fleet audit with the saved agent run outputs |
| `e226bdb1c56cea1405c6425d90189c836b0eef77` | `8ca5124c72c83d2e6d453fe307ac83b3c2a9ffae` | Add Q3 2026 to the Backblaze download list (ingest cap unchanged) |
| `5a93d864279acd94d99225338b1b50e849042c2c` | `f9867cff8464cbf36ceb499a833cb050936f5fbc` | Ingest Q3 2026 as the sealed split without moving earlier data |
| `f8bab86cdca5b559832d3125f0e0169a2d5a7e0b` | `94501a285fd26ebba84b9fc686c22a4fffc9466a` | Compute fleet-wide feature statistics on pre-sealed rows only |
| `edeb1b3cfc16ff48f4be581fe9959b579b5656cc` | `46bac80730e6396440ce6c9b12dc6cfcad86036e` | Train on Q1+Q2 2026 and test on Q3 2026, with figures per split |
| `b81a0ff4d86668199e3ea9d3c63ca0f07df7227c` | `6129dde73f3c247c18051bc528a5c746b08f45f8` | Retire the frozen model of record and drop its run id from the docs |
| `9dbadd84884829b5faa93f91c4f211e55d44a6e9` | `90e9006c330a63891569af030b63b2d3884e8d49` | Remove the retired model-of-record section from the runbook |
| `f950556986cdaea961e05c8e1e8ccc005dc0032c` | `80435b08109c869dd53169f3ba7a265d3da4d0df` | Compare label horizons on the same drives, matched on false-alarm rate |
| `e78c0d9112be261fbf1ccc097b36eee2d058dcd0` | `32d656fde20d0169872a34200a8f960cfe746df9` | Compare survival Stage 2 models in the cascade experiment |
| `6c61e35f1f8e4d4e35f4f4d1e21c335645f062ac` | `bde6ec2bd10618e35f5f36724ea958ee4a1a3e14` | Merge feat/q3-stage2: horizon comparison and survival Stage 2 |
| `3bfe60950fe71e749bb87e75128c0aa4dacef869` | `2a60ef54f7f412954d871a0c19fcd444dab10ec8` | Remove the Q4 2026 references |
| `e4e63250f4b69aaf1ad9135a537415a0decebc40` | `4345664028e4080d749fd29ab632bce4426de742` | Add the frozen bundles of the Q1+Q2 -> Q3 runs (30, 44 and 60 days) |
| `e60db0d898f2ac02d211e5fbf6ec009888e78fd7` | `26be96ad70ddc9c71f81de8bef00d2b764bdbc54` | Report the Q1+Q2 -> Q3 runs: horizons, cascade and survival Stage 2 |
| `db5eca3573b84926cdf5b959121926662e40d138` | `9981b12b48ccc7e677a5ccad95690cdf34fb66f2` | Update the docs for the Q1+Q2 -> Q3 findings (runbook 1.1.4) |
| `0f7b251f1abc35a70738eca7459dc25714503c72` | `c5d72e30ab489c93e1f95e2dca46fe8107a9b19b` | Embed the frozen runs' precision-recall and confusion plots in the runbook |
| `5a35f625722586f4cd84ff035d729253af6c8055` | `76ad12c08f23344e9fc4bd613f05d2cbf9c61276` | Add the cascade and horizon-comparison plots to the runbook |
| `1dc6a6cc99b687a34d1a6a871228a205f178b2df` | `4fb351a266c38941a91cd3d0252967e1d02d5522` | Track the results of the Q1+Q2 -> Q3 runs under data/audit |
| `0513ce574ed113202926ad1df804742429937721` | `044ee8025a1a40ca1f3f4f9c52eed228c104f62a` | Document data/audit and note that the Q3 run results are committed |
| `a73544a27c261ac765d0fc8e35f5ed7371103e90` | `ef5fb83a160811eb01568f2cc6afdae2668447d6` | Merge the Q1+Q2 -> Q3 2026 runs: horizons, survival Stage 2, cascade |
| `630b432e1d9a549b0dfb2a9c90528f918484bfbe` | `c31ea2b8efa1163bd93ec7c4e8bd17383e71de21` | Describe the frozen runs |
| `f2e18057a68879f35955ebed1b503320faef2771` | `c0699a362fe27326d61a298492025bf0a695b882` | Describe the current model c459046e in frozen_runs/README.md |
| `42c2a893e055d84558c7bf8d15c0f50bd8a24edb` | `a41f0cf1720995b6d9dced3f70b67bade2a91210` | Deep dive into the 30-day Q3 run; map the evidence folders in the runbook |
| `8dcca9bc81d0b607ea200700e662fd160d27d48d` | `8b5c1391c0f9d6b3f0d9bd4d8a0e3a7327e6ed64` | Cluster-based action policy: method, rule and tooling; list the Q3 commands |
| `7e1cb64adcfd7d7b06a6d076bbc027bcd5897591` | `2077976f362dad6b544c388a3f5dcb232a075a64` | Cluster action policy: results, decision and tracked outputs |
| `7055fafae1abbd59c4511d0bef9173937b69d11d` | `582b4058c45e2c90db67a42ee24cf42d2c1fee26` | Stacked Stage 2 (LightGBM + Cox): method, runs on Q3 and verdict |
| `9a3cd7e940815fb8596230d02fbf661a9bf5f827` | `6f355cd2cfbae96824955570ecf12d1e5347de55` | Move the stacked Stage 2 section to follow Section 1.1.4 |
| `cfa917a4460b09bca29a3a152c0ba953154c82c5` | `cdb8d18601d8e861e31220fb3e11a25a36f0abe7` | Stacked Stage 2: both-axes verdict, figures, frozen run ids; track the outputs |
| `a6222fc13f22033faf4b855b0c8e0244c6106898` | `a8d6d39c50b0fc940b5e612d5cf1e97fa892d2e7` | Track data/audit in full: drop its ignore rules |
| `508ac311e212c06e1c484d4ded57378858586aa3` | `167dbcc3b56a3d8459253a9dd28648c6201ce091` | Track audit logs that have no absolute paths; keep the others local |
| `bb65cd09bcc45fce3b6d06467c12c817878c8230` | `069c8ef9d89b02c034ade0215346a411559136fe` | Refuse commits that add an absolute path |
| `149938622d13de8c6b08cbe6b1c612c44af1407b` | `9754d99e133165ac19a3cb04fc6e46805cd07851` | Runbook 1.1.4a: drop an unsupported plot claim, add a matched-budget figure |
| `35ec8c96c8106798367872a2b3305d457d2f7dc8` | `adc96b781fb5a0864dab4908766047600a1dcaaf` | Merge the stacked Stage 2, cluster action policy and absolute-path check |
| `22e22cc85213d174773df064101e3f00df6926c2` | `fe3db3363eca819f1e638939e171a1e531e50ed7` | Report: centre the two-stage cascade; sync docs with the Q3 results |
| `f51d5f484da6d4529c0cb503eac6a1ed35ddae91` | `b8c6abc2512ea8a8b07df7432e3ae6ff568f820b` | Merge the report update for the Q3 results |
| `a593f1d5536b1bbd58424e795b7ee374f9866077` | `13b45018bda69121cb722d8cd7e6d525536ad7c8` | Design a Knowledge component for the MAPE-K agent and fix its screen rule |
| `d62a0f9480d16e371759cde3701b1319e37576b0` | `bf06648a59f28149b99142b77d036f5039e15901` | Phase S: freeze the stacked Stage 2 as a scorer; add cost-based tiers |
| `681f922659b309bbe9b559bb18ea57f6dbe2586e` | `72f7ae47c5498f0dcf1ecfa0468cab60a039ef78` | Freeze the 60-day stacked Stage 2 as a scorer (frozen_runs/stack_h60_seed0) |
| `e1617a2bae734a0b48fb72925539dc1d3b1a7eab` | `0e51dacaf3e290c0f94f85c7cf778571efba5f0e` | Knowledge-base design: record the screen's measured power (section 9) |
| `74a367fcc2a88ad5412a4da178acebfbcaca5c05` | `f26e0a679caeeffa1c10a2c1a023c6f58f881451` | Knowledge-base design: Amendment 1 to the screen rule, before any screening |
| `a51eb1bad2505246b033a16bc94e603d516a91e3` | `479ca87715bf504ac5c9adc89184c61cfe2c75df` | Phase 0: point-in-time options for validation-only, non-MLflow scorers |
| `47a7e7145cc195d906bc5aa6bd4c27ed1ec3ef2c` | `b70958740abf2b8e8f63927576a78980e0da6500` | Phase 0: make agent-pit-validation, the four-arm validation baseline |
| `089f238c0cf60e0469274f5db2858307fd5ebbd5` | `ff258e8d564dcad726a52ab1b84bb8843a5d0a91` | Phase 0 output: point-in-time validation baseline at 60 days (frozen) |
| `8bfab3d2aa9c948c70104274816472ddaeeb4e50` | `8fb7f7579eba414545c132f4ae02644de5fd052c` | Knowledge-base design: add a running progress log (section 11) |
| `116cf7be1f23b6dd18188dedc459a5f87ea83d50` | `7a53d7f755bb9c1e089a044109e4ea5ceeb8a66f` | Phase 1: the knowledge store (src/agent/knowledge/store.py) |
| `d35275a5cbde8fb4c7ec993b89b0e2e603ec762a` | `a2b1b07a3d3da7f15521c4a7385f7f5018e3ea0f` | Phase 1: knowledge builder, self-checks and command-line tool |
| `a0b77d28c0f1a3179401a26feac0673be64ba1ca` | `9c65f9f9f7086e7a60b4b4949cde388e840a99cb` | Knowledge-base design: progress log after the store, builder and checks |
| `90c12bb048b8157029b7bacf71e4568c7b71a939` | `fadc83b440fb30a10b2f14ffa7abeecb35631012` | Phase 1: wire the knowledge base into the MAPE-K loop (no decision change) |
| `27f8e098c8d64b67f254494d3f4056e7e04fb9e7` | `19a5a441b0b2e5d1b8761da90d66bd72423a27de` | Knowledge store: censoring is not a final outcome |
| `165e9c4611a74edb239c91fca22aff053f55eb63` | `3fe1ff7736ca0ff000e1cd21147733f63993eb10` | Knowledge-base design: progress log after the loop wiring and the censoring fix |
| `a9ba58b95e8f9ebb28b9820e11d382bdc37dd8e2` | `b4141ccb4a41d39e019227926ef02f9ce1355405` | Phase 2: knowledge rules and the pre-registered screen |
| `1266ed55bb7d540459502e304b8f471c1e83ac52` | `10ffc063fb8e980242a65bdf1feca7a722a7f3b1` | Knowledge screen: --describe-tier-rules, a descriptive look outside the verdict |
| `225aa9f4d5171ef81dafdf2a7438c0531d6e87bd` | `1376b1ba5afd3d0607603527061512fcaa7dd1a5` | Knowledge screen: wrap a long line (lint) |
| `c1da11d323913a7a7dbdbaa226343b3863c3fb9c` | `0043d8bf795a21ab68704767b93ccbcaa9c1aba3` | Phases 1-2 output: knowledge store checks and the rule screen (frozen) |
| `c93f6c3d707257aba3ac7bbdf0c556da1e85cdee` | `47a52e9a91a946d091b366b86c7442c9224cf8ad` | Knowledge component: record the results in the runbook, design doc and ADR 0004 |
| `c7cceedfc1b0c6e89e78f7ba73a7fc55dfa2c4b2` | `6c09327384c66bae4960c41fc8ab52f781f5e837` | Knowledge store: pack and unpack it as a tracked gzip; make targets |
| `afe4166a13fe6f14e259e6f8ae1e73b92ef6ac1c` | `a6031ecc5c8076c89f2eb85d4bbba915e8aa0ea2` | Track the knowledge store as a gzip and document how to restore it |
| `793401492a2b3ad374e374d15d472e96ee8320f3` | `946a181543bd44d8abaed3de5802b6143277aa9f` | Write repo-relative paths in logs: scrubber and logging hooks |
| `1c63e9b2c0e2775ca4bec9498b557a00b09fa5eb` | `ca769fd810684574359bbd17ab0b83e03b8e6f15` | Track the run logs under data/audit with repo-relative paths |
| `0f073969ad4af8e6b770c65dcc2238147a990c8f` | `0046c3f3bb4e602856d0c05c844237429a14745a` | Merge the Knowledge component for the MAPE-K agent |
