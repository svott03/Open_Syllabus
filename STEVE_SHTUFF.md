# Data & Recap
I added you as owner for the project. You should see an invite over gmail at svott03.

All syllabi are in storage bucket syllabus-480300-syllabi. Metadata is also there in the syllabi_index.csv file at the top level of the bucket. Syllabi are stored in a way that should be captured by syllabi_index's local_path but maybe do a sanity check first? Overall it's 146.6k files spanning 56.9GB of data so tread lightly, but lowk the budget is like $2k for compute. If you wanna use any service not in the GCP system or if I'm about to get hella budget alerts just ping over discord.

To recap, we need to provide clients with the ability to filter by school and topic (i.e. department) and get all referenced titles in descending order. We also need to, given a title is selected, get all coreferenced titles in descending order, also being able to filter this in descending order. It seems that we may also need to add further comparisons, such as the breakdown of references by topic, but that our analysis will be pretty strictly limited to per-administration, so like tops 1M documents. So your call on the data structs we use. Obviously coreference is still the bottleneck.

# What is a topic
A topic refers to something like "Law", "Political Science", etc. Unlike OSP, we don't need to guess. https://syllabi.osu.edu/student/, https://syllabus.ufl.edu/, https://utdirect.utexas.edu/apps/student/coursedocs/nlogon/ are the links for the syllabi sites. Have Claude or some other clanker help you build a mapping from the csv fields and course prefixs (CSCI type) to topics, and then a further mapping / reduction for redundancies / overlap between the schools is probably the fastest approach. Each syllabus should only have 1 topic, or none if none can be found. Users filtering should be able to select multiple topics when querying and get the union. The OSU and UTAustin syllabi all came from one archive, whereas UFL had much sloppier data, split into present year on one platform and all prior years in like 20 diff formats.

# What is a syllabus
Syllabi have been deduped by links and obvious redundancies, but there are still cases with multiple syllabi per year (if 3 instructors taught the same course for ex) or per instructor-course pair (if taught over multiple years). Your call but deduping may not be stupid. Especially bc we have so much fucking data. Still, your call. Much more important to demo than to demo high quality. Our client isn't any of these schools.

# What is a title
A title needs to have an actual title or name and an author or set of authors. We can compress multiple versions or editions to 1 entry in our DB. Obviously the PDF -> Citations part is the hardest. https://github.com/grobidOrg/grobid It seems like there may be existing software, OS or not, that's useful, but I tested that link and it didn't really work. Idk shop around and try to have some fallbacks. Lowk we can ball out so a (PDF -> related works section -> ask AI to guess title fields from related works -> pass that to some open source or title matcher) could be a good pipeline. Or we try non-AI and fallback to AI. In the contract I scoped 100 manually annotated samples, and I have some friends willing to annotate them, so just lmk when / if you need these.

# Final deliverable

API endpoints (or tool integration for elasticsearch or graphdb or smn) for the following
- GET list of all topics (to filter by)
- GET / search titles (maybe elasticsearch as we want that crispyness when typing something in)
- GET descending (co-)reference counts 
  - params:
    - schools: None | some set of "ufl", "osu", "utaustin". None behavior is as though all selected
    - topics: None | some subset. None behavior is as though all selected (but obv on query side use intelligent approach) 
    - title: None | some title. We need either title IDs or slugs for this, and None means we're just counting over all syllabi whereas if selected it's syllabi where the chosen title is already selected
  - should return the descending (title, count) pairs

# After this

We'll lowk also need to add downloading the data, getting the breakdown of reference by topic for a given title or title/school combo, and probably other wackadoodle shit but for now who gives a fuck. Let's tunnel vision on this.