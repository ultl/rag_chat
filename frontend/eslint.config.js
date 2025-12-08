const next = require('eslint-config-next')

module.exports = [
  ...next,
  {
    ignores: ['dist/**', 'coverage/**']
  }
]
