<!-- Demo: https://github.com/PaddlePaddle/PaConvert/pull/71 -->
### PR APIs
<!-- APIs what you've done -->
```bash
torch.dist
torch.abs
torch.transpose
torch.addcmul
```
### PR Docs
<!-- Describe the docs PR corresponding the APIs -->
the docs PR corresponding the APIs is as follows:
PR: https://github.com/PaddlePaddle/docs/pull/xxx
### Description
<!-- Describe what you've done -->
#### A total of 4 translatedd APIs.
The corresponding configuration file is written in PaConvert/paconvert/api_mapping.json and the corresponding writing policy is in PaConvert/paconvert/api_matcher.py, the basic correspondence can be found as follows.
* first-class APIs
 ```bash
torch.dist-->GenericMatcher
```
* second-class APIs
 ```bash
torch.abs-->GenericMatcher
 ```
* third-class APIs
 ```bash
torch.transpose-->TransposeMatcher
 ```
* fourth-class APIs
 ```bash
torch.addcmul-->AddCMulMatcher
 ```
#### others
* add pull request template in .github/PULL_REQUEST_TEMPLATE.md
