# Robot Learning Collective website

Static site for the Robot Learning Collective, built for GitHub Pages using the Minimal Jekyll theme.

Push changes to `main`. GitHub Pages (Pages/Source set to the `main` branch) will build and publish automatically.

## Local development

### Prerequisites

Install Ruby and Bundler (Ubuntu/Debian):

```bash
sudo apt install -y ruby-bundler ruby-dev build-essential
```

### First-time setup

Configure Bundler to install gems locally (avoids needing root permissions):

```bash
bundle config set --local path 'vendor/bundle'
```

Install dependencies:

```bash
bundle install
```

This creates a `vendor/bundle` directory with all required gems. The directory is git-ignored.

### Running the site

Start the local Jekyll server:

```bash
bundle exec jekyll serve
```

The site will be available at **http://localhost:4000**. Jekyll watches for file changes and rebuilds automatically.

Press `Ctrl+C` to stop the server.
