- [ ] Tests added / passed

  ```bash
  $ py.test --flake8 -v -s .
  ```

  If the flake8 tests fail, the quickest way to correct
  this is to run `autopep8` and then `flake8`
  to fix the remaining issues.

  ```
  $ pip install -U autopep8 flake8
  $ autopep8 -r -i .
  $ flake8 .
  ```
