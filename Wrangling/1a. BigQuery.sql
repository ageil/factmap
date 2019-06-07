SELECT
  id,
  created_utc,
  retrieved_on,
  subreddit_id,
  url,
  title,
  subreddit,
  'domain',
  is_self,
  subreddit_subscribers,
  num_comments,
  author
FROM
  `pushshift.rt_reddit.submissions`
WHERE is_self = false