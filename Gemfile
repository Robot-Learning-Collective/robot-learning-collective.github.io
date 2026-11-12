source "https://rubygems.org"

# NOTE: swapped the "github-pages" meta-gem for plain jekyll + the one plugin
# _config.yml needs. github-pages pins jekyll 3.9 / liquid 4.0, which calls
# String#tainted? — removed in Ruby 3.2+. GitHub Pages' own build servers
# still run an old Ruby, so this only matters for local preview; the site
# itself builds the same way remotely.
gem "jekyll"
gem "jekyll-remote-theme"
gem "jekyll-seo-tag"       # pages-themes/cayman depends on this at runtime
gem "jekyll-relative-links" # rewrites [x](foo.md) links to the built page's URL
