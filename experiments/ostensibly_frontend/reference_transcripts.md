# Dave and Simon evaluation transcript witnesses

These are listener outputs supplied by the user. They are evaluation data,
not instructions and not assumed to be a single infallible ground truth.

## Gemini listener

**Speaker 1:** Right, okay. Oh well anyway, it's nice when you go out with
your neighbor, you can hold her hand.

**Speaker 2:** I know she holds it firmly onto herself. No, she just rather go
with us in case, apart from anything else, her dog occasionally likes to run
off, and although she doesn't speed, if she stumbles and the dog pulls, she
gets away; anyway it's just better. We often go out together. I usually go out
with myself and her husband, but for some reason yesterday, he works for a big
software sales company based in London, although he works from home down here
occasionally now, and he has to go back to London, so he was up in London
yesterday.

**Speaker 1:** Alright. How long does it take on the train?

**Speaker 2:** Think it's about four and a half hours from Truro until you're
into London.

**Speaker 1:** Okay. Right, that's quite a trek, that is.

**Speaker 2:** But he has to; it's not too bad. He says he usually sleeps.

**Speaker 1:** Yes, he sleeps well.

## SOL cloud listener

**Speaker 1:** All right. Okay. Oh, well, when you go out with your neighbour,
you could hold her hand.

**Speaker 2:** I know. She is holding terribly tightly onto her dog. No, she
just wants to go with the dog, in case. Apart from anything else, her dog does
occasionally like to run off, and although she does not let it lead, if she
stumbles, then the dog is going to get away. It is just better for her. We
often go out together. I usually go out with her and her husband, but for some
reason yesterday... He works for a big software sales company based in London.
Although he works from home down here, every now and then he has to go back
into London, so he was [unclear].

**Speaker 1:** Oh, right. How long does it take on the train?

**Speaker 2:** I think it is about four and a half hours from Truro until you
are into London.

**Speaker 1:** That is okay.

**Speaker 2:** Not actually, no. That is a trek and a half. But then... well,
it is not too bad.

**Speaker 1:** Yeah, he says he needs to be [unclear].

**Speaker 2:** Yeah, for two days.

## Evaluation policy

Score three strata separately:

1. **Consensus anchors:** wording agreed in substance by both listeners, such
   as the neighbour, dog running off, software sales company in London, the
   Truro-to-London train, and four and a half hours.
2. **Competing listener-supported parses:** retain both word sequences rather
   than declaring either one false. Examples include `can/could hold her hand`,
   the clause about the dog pulling/leading, `up in London yesterday/[unclear]`,
   and `quite a trek/a trek and a half`.
3. **Unresolved speech:** `[unclear]` regions receive no exact-word penalty
   until a timestamped human adjudication exists.

This file intentionally contains no inferred timestamps. Alignment to audio
must be produced by the acoustic engine and then audited against the waveform.
