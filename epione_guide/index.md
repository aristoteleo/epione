# Documentation

## EpiOne: Generative cartography of single-cell epigenomes

EpiOne provides tools for generative modeling and analysis of single-cell epigenomic data, with a focus on intuitive workflows and clear APIs.

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Installation {octicon}`plug;1em;`
:link: installation
:link-type: doc

New to EpiOne? Start with the installation guide.
:::

:::{grid-item-card} User guide {octicon}`info;1em;`
:link: user_guide/index
:link-type: doc

Learn core concepts and how pieces fit together.
:::

:::{grid-item-card} API reference {octicon}`book;1em;`
:link: api/index
:link-type: doc

Explore the full EpiOne Python API.
:::

:::{grid-item-card} Tutorials {octicon}`play;1em;`
:link: tutorials/index
:link-type: doc

Hands-on examples using real data.
:::

:::{grid-item-card} Discussion {octicon}`megaphone;1em;`
:link: https://github.com/aristoteleo/epione/discussions

Questions or ideas? Join the conversation.
:::

:::{grid-item-card} GitHub {octicon}`mark-github;1em;`
:link: https://github.com/aristoteleo/epione

Report issues or contribute on GitHub.
:::
::::

```{toctree}
:hidden: true
:caption: Getting started
:maxdepth: 1

installation
gpu
user_guide/index
tutorials/index
```

```{toctree}
:hidden: true
:caption: Bulk tutorials
:maxdepth: 2
:titlesonly: true

tutorials/bulk/upstream
tutorials/bulk/atac
tutorials/bulk/hic
```

```{toctree}
:hidden: true
:caption: Single-cell tutorials
:maxdepth: 2
:titlesonly: true

tutorials/single/basics
tutorials/single/clusters
tutorials/single/regulation
tutorials/single/footprints
tutorials/single/scenic
tutorials/single/multimodal
tutorials/single/spatial
tutorials/single/translation
tutorials/single/hic
```

```{toctree}
:hidden: true
:caption: Reference
:maxdepth: 1

api/index
faq
changelog
references
developer
```

```{toctree}
:hidden: true
:caption: Community
:maxdepth: 1

Discussion <https://github.com/aristoteleo/epione/discussions>
GitHub <https://github.com/aristoteleo/epione>
```
