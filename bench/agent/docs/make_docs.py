#!/usr/bin/env python3
"""Writes the ask-docs document set into this directory: 24 synthetic personal documents, two
files the skill must never read (a hidden note and a PDF, which hold the answers to two of the
questions nothing else answers), and questions.json with the planted facts.

    python3 bench/agent/docs/make_docs.py           # (re)write the files
    python3 bench/agent/docs/make_docs.py --check   # exit 1 unless the committed files are exactly
                                                    # what this writes, every planted quote occurs
                                                    # exactly once in its file, and the mix holds

Everything is fixed text or built from fixed lists, so the output never changes. People, places
and companies are invented. Near-miss distractors (an older lease, an older status update, the car
insurance, the home router) make ranking alone not enough.
"""

import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
KB = 1024


def lease_alder():
    return """# Residential Lease Agreement

**Property:** 42 Alder Street, Apartment 3B, Riverton
**Landlord:** Harbor Point Properties LLC, 900 Quay Road, Riverton
**Tenant:** Jordan Avery
**Term:** 1 September 2026 to 31 August 2027 (twelve months)

This agreement is between the landlord and the tenant named above for the apartment named above.
The tenant has inspected the apartment and accepts it in its present condition, except for the
items listed on the move-in checklist signed on 28 August 2026.

## 1. Rent

Monthly rent: $1,850, due on the first day of each month.
Rent is paid by bank transfer to the account named in Schedule A, or by cheque left in the office
drop box at 900 Quay Road. Cash is not accepted.
If rent is not received by the fifth day of the month, the tenant pays a late fee of $75.
A returned payment costs $35 in addition to any late fee.

## 2. Security deposit

The tenant paid a security deposit of $3,700 on signing. The landlord keeps it in a separate
account and returns it within 21 days after the tenant moves out, less the cost of any damage
beyond normal wear and tear, with a written list of deductions.

## 3. Utilities

Water, sewer and trash collection are included in the rent. The tenant pays for electricity,
gas, internet and any streaming or cable services, and must put the electricity account in the
tenant's own name before the start of the term.

## 4. Pets

One cat is allowed with a pet deposit of $300. Dogs, rabbits and birds are not allowed without
the landlord's written consent. The tenant is responsible for any damage a pet causes and for
keeping shared hallways free of litter and pet hair.

## 5. Maintenance and repairs

The tenant reports needed repairs through the Harbor Point tenant portal or by phone on
555-0142. Emergencies (no heat, a water leak, a broken entry door) are answered within four
hours, day or night. Routine repairs are scheduled within seven days. The tenant replaces light
bulbs and smoke alarm batteries and keeps drains clear.

## 6. Entry

The landlord may enter the apartment for inspections and repairs with 24 hours' notice, between
9:00 and 18:00 on weekdays. No notice is needed in an emergency.

## 7. Renewal and notice

Either party may end the lease at the end of the term by giving 60 days' written notice. If
neither party gives notice, the lease continues month to month on the same terms, and either
party may end it with 30 days' written notice.

## 8. Parking and storage

The tenant has one parking space, number 14, in the rear lot, and storage cage 3B in the
basement. Bicycles are kept in the storage cage or the bicycle room, never in hallways.

## 9. Quiet hours

Quiet hours are 22:00 to 7:00 every day. Parties need the neighbours' agreement in advance.

Signed by Jordan Avery (tenant) and Dana Whitlock for Harbor Point Properties LLC
(landlord) on 28 August 2026.
"""


def lease_maple():
    return """# Lease Agreement (2024)

**Property:** 17 Maple Avenue, Unit 2, Riverton
**Landlord:** Okonkwo Family Rentals
**Tenant:** Jordan Avery
**Term:** 1 June 2024 to 31 May 2025

The tenant moved out of this unit on 31 May 2025. This copy is kept for the records.

## Rent

Monthly rent: $1,450, due on the first day of each month.
Rent is paid by cheque or bank transfer. If rent is not received by the third day of the month,
the tenant pays a late fee of $50.

## Deposit

The security deposit was $1,450. It was returned in full on 18 June 2025.

## Utilities

The tenant pays for electricity and gas. Water and trash collection are included.

## Pets

No pets are allowed in Unit 2.

## Repairs

Repairs are requested by phone from the landlord, Mrs. Ada Okonkwo, who lives in Unit 1. The
furnace is serviced every October. The tenant clears snow from the front steps.

## Notice

Either party may end the lease at the end of the term with 30 days' written notice.

## Parking

Street parking only. A residents' permit from the city costs $60 a year.

## Move-out

At move-out the tenant cleans the unit, returns both keys and the garage remote, and gives a
forwarding address for the deposit. The landlord inspects the unit with the tenant present.

Signed by Jordan Avery and Ada Okonkwo on 20 May 2024.
"""


def trip_lisbon():
    days = [
        ("Friday 23 October", "Arrive at 07:05, take the metro to Baixa-Chiado, leave bags at the hotel. Walk the "
         "Alfama and have lunch at the tasca on Rua dos Remedios. Early night."),
        ("Saturday 24 October", "Tram 28 in the morning before the crowds, Sao Jorge castle, sunset at the "
         "Miradouro da Graca. Dinner booked at Casa Lume, 20:30."),
        ("Sunday 25 October", "Day trip to Sintra: train from Rossio station at 09:11. Pena Palace tickets are "
         "for the 11:00 slot. Back by 18:00."),
        ("Monday 26 October", "Belem: the monastery, the tower and custard tarts at the old bakery. Afternoon at "
         "the tile museum."),
        ("Tuesday 27 October", "Free day. Maybe the beach at Cascais if the weather holds, otherwise the "
         "Gulbenkian museum and garden."),
        ("Wednesday 28 October", "Food tour at 11:00, meeting point Praca da Figueira. Pack in the evening."),
    ]
    plan = "\n\n".join(f"**{day}.** {text}" for day, text in days)
    return f"""# Lisbon trip, 22 to 29 October 2026

## Flights

Outbound flight GL 218 departs Newark at 18:40 on Thursday 22 October and lands in Lisbon at
07:05 the next morning. Seats 23C and 23D. Check in online 24 hours before.

Return flight GL 219 leaves Lisbon at 10:15 on Thursday 29 October and lands in Newark at 13:40
local time.

Gull Air booking reference: QX7P2L. One checked bag each, 23 kg.

## Hotel

Casa do Rio, Rua das Flores 12, Lisbon. Check-in from 15:00, but they keep bags from the morning.
Breakfast is included. Booking number 448190, paid in full.

## Plan by day

{plan}

## Budget

Flights $1,240 for two, hotel $980 for six nights, about $90 a day for food, transport and
tickets. Get a Viva Viagem card at the airport for the metro and trams.

## Before leaving

Ask Sam to feed Miso twice a day. Put the mail on hold from 22 to 30 October. Charge the travel
adapters. Download offline maps. Tell the bank about the trip.
"""


def washer_manual():
    safety = [
        "Read all instructions before using the washer.",
        "Do not wash items soaked in petrol, solvents or other flammable liquids.",
        "Keep children and pets away from the washer while it runs and check the drum before every use.",
        "Do not climb or sit on the washer, and do not lean on the open door.",
        "Unplug the washer before cleaning it or opening the service panel.",
        "Do not install the washer outdoors or where it can freeze.",
        "Use only the new hose set supplied; do not reuse old hoses.",
        "Plug the washer into a grounded outlet of its own; do not use an extension cord.",
        "Do not touch the plug with wet hands.",
        "Keep the area under and around the washer free of lint, dust and packaging.",
        "Do not open the detergent drawer while the washer is filling.",
        "If the power cord is damaged, have it replaced by a qualified service technician.",
    ]
    cycles = [
        ("Cotton", "40 °C", "1,200 rpm", "2 h 30 min", "everyday cotton, towels and bed linen", "8 kg",
         "Wash towels and bed linen at 60 °C once a month."),
        ("Cotton Eco", "40 °C", "1,400 rpm", "3 h 20 min", "normally soiled cotton, using the least energy",
         "8 kg", "The long time is how it saves energy: the water is heated less and the clothes soak longer."),
        ("Synthetics", "40 °C", "1,000 rpm", "1 h 50 min", "polyester, nylon and mixed fibres", "4 kg",
         "Turn printed shirts inside out."),
        ("Mixed Load", "30 °C", "1,000 rpm", "1 h 20 min", "cotton and synthetics washed together", "4 kg",
         "Separate new dark items for their first few washes."),
        ("Delicates", "30 °C", "600 rpm", "50 min", "lace, viscose and fine fabrics", "2 kg",
         "Use a mesh bag for underwear and anything with hooks."),
        ("Wool", "30 °C", "800 rpm", "45 min", "machine-washable wool and cashmere", "2 kg",
         "Use wool detergent and dry the items flat."),
        ("Quick 15", "20 °C", "800 rpm", "15 min", "small, lightly soiled loads up to 2 kg", "2 kg",
         "Use a third of the usual detergent dose."),
        ("Sportswear", "30 °C", "800 rpm", "1 h 10 min", "technical fabrics and gym clothes", "3 kg",
         "Do not use fabric softener; it blocks breathable fabrics."),
        ("Bedding", "40 °C", "800 rpm", "1 h 40 min", "duvets and pillows that fit the drum", "one duvet",
         "Check the care label; feather duvets may need a professional clean."),
        ("Baby Care", "60 °C", "1,200 rpm", "2 h 45 min", "baby clothes, with an extra rinse", "4 kg",
         "Use a detergent for sensitive skin."),
        ("Rinse and Spin", "cold", "1,200 rpm", "20 min", "rinsing a load again", "8 kg",
         "Useful after hand washing, or to add softener."),
        ("Drum Clean", "90 °C", "400 rpm", "1 h 30 min", "cleaning the empty drum", "empty",
         "Leave the door open afterwards so the drum dries."),
    ]
    parts = [
        ("door lock", "Close the door firmly and restart the cycle."),
        ("water inlet valve", "Turn the tap fully on and check that the inlet hose is not kinked."),
        ("heater", "Restart the cycle; if the code returns, call service."),
        ("temperature sensor", "Restart the cycle; if the code returns, call service."),
        ("motor", "Reduce the load and restart; if the code returns, call service."),
        ("pressure sensor", "Unplug the washer for one minute and restart."),
        ("detergent pump", "Check the detergent drawer is closed and restart."),
        ("control board", "Unplug the washer for five minutes and restart."),
        ("drum speed sensor", "Check the transit bolts are removed and the load is balanced."),
        ("door switch", "Close the door firmly; if the code returns, call service."),
    ]
    kinds = ["did not respond", "reported an open circuit", "reported a short circuit",
             "timed out"]
    codes = []
    for n in range(1, 41):
        if n == 21:
            codes.append("E21: The drum did not drain within 10 minutes. Check the drain hose for kinks "
                         "and clean the drain pump filter.")
            continue
        part, fix = parts[(n - 1) % len(parts)]
        kind = kinds[(n - 1) // len(parts)]
        codes.append(f"E{n:02d}: The {part} {kind}. {fix}")
    trouble = [
        ("The washer does not start.", "Check that the plug is in, the door is closed and the child lock "
         "is off. Press Start/Pause and hold it for one second."),
        ("The washer does not fill.", "Open the tap fully. Clean the small filters at the inlet end of the "
         "hose. Make sure the hose is not kinked or frozen."),
        ("Water stays in the drum.", "The drain hose may be kinked or set too high. Look up any code on "
         "the display in the error code list."),
        ("The washer shakes or walks.", "Make sure the four feet touch the floor and the lock nuts are "
         "tight. Remove all transit bolts. Wash small loads with a few larger items."),
        ("Clothes are still very wet.", "The load may be unbalanced, so the washer reduced the spin. Add or "
         "remove items and run Rinse and Spin."),
        ("There is too much foam.", "Use less detergent, and only detergent marked for front-load "
         "machines. Soft water needs less detergent."),
        ("The door will not open.", "The door stays locked while there is water in the drum or the drum "
         "is hot. Wait two minutes after the cycle ends."),
        ("There is a smell from the drum.", "Run the Drum Clean cycle, wipe the door seal and leave the "
         "door and the detergent drawer open between washes."),
        ("Detergent stays in the drawer.", "Clean the drawer and its siphon cap. Check the water pressure "
         "and that the tap is fully open."),
        ("The cycle takes longer than shown.", "The washer adds time for unbalanced loads, cold inlet "
         "water or extra foam. This is normal."),
        ("Water leaks under the washer.", "Check that the hoses are tight at both ends and the drain pump "
         "filter cap is closed. Call service if the leak continues."),
        ("The display shows a flashing key.", "The child lock is on. Hold Temp and Spin together for three "
         "seconds to turn it off."),
        ("Clothes come out creased.", "Use a lower spin speed, a smaller load or the Easy Iron option."),
        ("The washer makes a rattling noise.", "Coins or hair clips may be in the drum or the drain pump "
         "filter. Empty the pockets before washing."),
        ("The display is blank.", "Check the fuse and the outlet with another appliance. The display "
         "turns off after 10 minutes without use; turn the dial to wake it."),
        ("The washer stops during the cycle.", "Start/Pause may have been pressed, or the door was opened "
         "with Delay End running. Press Start/Pause to continue."),
        ("Grey marks appear on clothes.", "Too little detergent, or fabric softener poured straight onto "
         "the clothes. Run Drum Clean and use the right dose."),
        ("The washer beeps at the end.", "This is the end signal. Hold Spin for three seconds to turn the "
         "sound off or on."),
        ("Steam does not appear.", "Steam is only on Cotton, Synthetics and Baby Care, and not with "
         "loads over 4 kg."),
        ("Water drips from the detergent drawer.", "Too much detergent or a blocked siphon cap. Clean "
         "the drawer and use less detergent."),
    ]
    cycle_text = "\n\n".join(
        f"{name}\n  Highest temperature {temp}, spin up to {spin}, about {time}.\n  For {use}."
        f"\n  Largest load: {load}.\n  Tip: {tip}"
        for name, temp, spin, time, use, load, tip in cycles)
    trouble_text = "\n\n".join(f"Problem: {p}\nWhat to do: {f}" for p, f in trouble)
    stains = [
        ("Coffee and tea", "rinse in cold water at once, then wash at the highest temperature the label allows"),
        ("Red wine", "blot, cover with salt or rinse with cold water, then pre-treat with stain remover"),
        ("Grass", "rub in liquid detergent and leave for 15 minutes before washing"),
        ("Blood", "rinse in cold water only; hot water sets the stain"),
        ("Grease and oil", "dab with washing-up liquid, leave for 10 minutes, wash at 40 °C or more"),
        ("Ink", "dab with rubbing alcohol on a cloth from the outside in, then wash"),
        ("Chocolate", "scrape off what you can, rinse from the back with cold water, then pre-treat"),
        ("Make-up", "rub in liquid detergent, then wash; for waterproof make-up use a little oil first"),
        ("Mud", "let it dry, brush it off, then wash as usual"),
        ("Sweat marks", "soak in warm water with a spoon of baking soda for an hour, then wash"),
        ("Tomato sauce", "rinse from the back with cold water, pre-treat, and dry in sunlight"),
        ("Candle wax", "freeze the item, chip off the wax, then iron between paper towels"),
    ]
    stain_text = "\n".join(f"{s}: {how}." for s, how in stains)
    faq = [
        ("Can I add laundry after the cycle starts?", "Yes, during the first 10 minutes. Press "
         "Start/Pause, wait until the door unlocks, add the item and press Start/Pause again."),
        ("Why is there water in the drum when it is new?", "Every washer is tested at the factory, and "
         "a little water stays behind. It is clean and harmless."),
        ("Can I use washing pods?", "Yes. Put the pod at the back of the empty drum, before the laundry, "
         "never in the detergent drawer."),
        ("How do I wash trainers?", "Remove the laces and insoles, put the trainers in a pillowcase with "
         "two towels, and use the Sportswear cycle at 30 °C with a low spin."),
        ("Can I wash a rug or a bath mat?", "Only if it has no rubber backing and fits the drum loosely. "
         "Use the Bedding cycle."),
        ("Why does the time on the display jump?", "The washer adjusts the time after it weighs the "
         "load and again after it checks the balance before the spin."),
        ("Is it safe to use the washer at night?", "Yes, but use Delay End so the spin does not run while "
         "you sleep, and make sure the tap and the hoses are in good order."),
        ("How long does the washer last?", "With the maintenance in section 11, about 10 to 12 years of "
         "household use."),
        ("Can I wash a down jacket?", "Yes, on Wool or Delicates with down detergent and a low spin. Dry it "
         "with two clean tennis balls in a tumble dryer so the down does not clump."),
        ("What does the drum light do?", "It turns on for two minutes when you open the door, so you can "
         "check for forgotten socks."),
        ("Can I turn off the end signal for good?", "Yes: hold Spin for three seconds when the washer is "
         "on. The setting stays until you change it again."),
        ("Does the washer need hot water?", "No. It heats its own water and must be connected to the cold "
         "supply only."),
        ("How much water does a cycle use?", "Between 40 and 60 litres for a full Cotton load, less for "
         "smaller loads and the short cycles."),
        ("Why does the washer pause before the spin?", "It turns the drum slowly to spread the laundry "
         "evenly. With a single heavy item it may try several times and then spin more slowly."),
    ]
    faq_text = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in faq)
    return f"""AQUAWAVE WM-7 FRONT-LOAD WASHER
OWNER'S MANUAL

Thank you for choosing the AquaWave WM-7. Keep this manual with the washer. Write the model and
serial number from the label inside the door here, so you have them ready when you call service.

Model: WM-7    Serial number: ____________    Date of purchase: ____________


1. SAFETY

{chr(10).join("- " + line for line in safety)}


2. INSTALLATION

Unpacking. Remove all packaging, including the foam block under the drum. Keep the transit bolts
and their plastic spacers in case you move house: the drum must be locked again for transport.

Transit bolts. Remove the four transit bolts on the back panel with the supplied spanner and
cover the holes with the plastic caps. A washer run with its transit bolts in place is damaged,
and the warranty does not cover that damage.

Levelling. Place the washer on a firm, flat floor. Turn the four feet until the washer is level
from side to side and from front to back, then tighten the lock nuts against the cabinet. A
washer that is not level shakes and makes noise during the spin.

Water supply. Connect the inlet hose to a cold water tap with a 3/4 inch thread. Tighten by hand
only. The water pressure must be between 0.1 and 1 MPa. Open the tap and check for leaks.

Drain. Hook the drain hose over a standpipe or the edge of a sink, between 60 and 100 cm above
the floor. Do not push the hose more than 15 cm into the standpipe, and do not seal it: air must
be able to enter, or the washer siphons.

Electrical. The washer needs a grounded 220 to 240 V outlet with a 10 A fuse. Do not use an
adapter or a power strip.

First wash. Run the Cotton cycle at 60 °C with half a dose of detergent and no laundry, to
rinse out any water left from factory testing.


3. QUICK START

1. Sort the laundry by colour, fabric and how dirty it is, following the care labels.
2. Load the drum loosely: you should be able to put your hand in above the laundry.
3. Add detergent to compartment II and softener to compartment III if you use it.
4. Turn the dial to a cycle. Change the temperature or spin if you need to.
5. Press Start/Pause. The door locks and the display shows the time left.
6. When the display shows End and the door unlocks, take the laundry out promptly.


4. LOADING THE WASHER

Empty all pockets. Close zips and hooks, tie strings and belts, and turn jeans and printed items
inside out. Wash small items such as socks and underwear in a mesh bag. Do not overload the
drum: clothes need room to move to get clean, and an overloaded drum also spins less. Mix large
and small items so the drum can balance before the spin. Wash new coloured items separately the
first few times.


5. CARE LABEL SYMBOLS

A tub with a number: the highest water temperature in °C.
A tub with one line under it: a gentler cycle, such as Synthetics.
A tub with two lines under it: a very gentle cycle, such as Delicates or Wool.
A tub with a hand: hand wash, or the Wool cycle at 30 °C.
A crossed-out tub: do not wash in water.
A triangle: bleach allowed. A crossed-out triangle: no bleach.
A square with a circle: tumble drying allowed. Dots show the heat.


6. CONTROL PANEL

The cycle dial selects the wash program. The display shows the time left, the temperature, the
spin speed and any error code. Temp and Spin change the defaults of the selected cycle. Delay
End sets when the cycle finishes, up to 24 hours ahead. Start/Pause starts the cycle or pauses it
to add a forgotten item. Holding Temp and Spin together for three seconds turns the child lock
on or off.


7. WASH CYCLES

{cycle_text}


8. OPTIONS

Prewash adds a short wash before the main wash, for heavily soiled loads; put detergent in
compartment I.
Extra Rinse adds one rinse, for sensitive skin or soft water.
Easy Iron reduces the spin and adds a slow tumble at the end to reduce creases.
Steam adds a burst of steam before the final spin, on Cotton, Synthetics and Baby Care only.
Delay End finishes the cycle at the time you choose.
Spin Off leaves the water out of the last spin, for items that must not be spun.


9. DETERGENT AND SOFTENER

Use only low-foam detergent marked for front-load machines. Compartment I is for prewash
detergent, compartment II for main wash detergent and compartment III for fabric softener. Do
not fill past the MAX line. Liquid detergent goes in compartment II with the liquid insert in
place. For small loads and soft water use half the dose printed on the detergent pack.


10. ERROR CODES

{chr(10).join(codes)}

If a code is not in this list, or a code returns after you followed the steps, unplug the washer
and call AquaWave service.


11. CLEANING AND MAINTENANCE

Drain pump filter. Clean the drain pump filter every three months. Open the small panel at the
bottom right of the front, put a shallow tray and a towel underneath, pull out the emergency
drain hose and let the water run out. Then turn the filter cap anticlockwise, remove lint, coins
and hair clips, rinse it and screw it back firmly.

Drum. Run the Drum Clean cycle once a month with the drum empty, or with a washing machine
cleaner.

Door seal. Wipe the door seal and the glass after each wash and leave the door open so the drum
can dry.

Detergent drawer. Pull the drawer out, press the release tab and rinse it under warm water once
a month.

Inlet filters. Once a year, turn off the tap, unscrew the inlet hose and clean the small mesh
filter with a soft brush.


12. TROUBLESHOOTING

{trouble_text}


13. STAIN GUIDE

Treat stains before they dry, and always test a stain remover on a hidden seam first.

{stain_text}


14. QUESTIONS AND ANSWERS

{faq_text}


15. SAVING ENERGY AND WATER

Wash full loads where you can: the washer weighs the load and uses less water for small loads,
but two half loads still use more than one full load. Most everyday laundry comes clean at 30 or
40 °C. Cotton Eco uses the least energy for cotton, although it takes the longest. Use Delay End
to run the washer when your electricity is cheaper. A high spin speed saves energy if you use a
tumble dryer afterwards.


16. MOVING THE WASHER

Unplug the washer, turn off the tap and drain the washer through the drain pump filter. Fit the
four transit bolts and their spacers again before moving it. Move it upright, and do not lift it
by the door or the detergent drawer.


17. ACCESSORIES AND SPARE PARTS

Genuine AquaWave parts are sold by AquaWave service and approved dealers: inlet and drain hose
sets, the drain pump filter cap, the detergent drawer and its liquid insert, door seals, the
stacking kit for a matching dryer, and the anti-vibration pads. Use only genuine parts: other
hoses may not hold the water pressure, and other seals may leak. Washing machine cleaner,
descaler and mesh laundry bags are sold in packs of three.


18. SERVICE

Before you call, look up any code in section 10 and try the steps in section 12. When you call,
have the model and serial number ready from the label inside the door, and the date of purchase.
Service visits are booked on weekdays between 8:00 and 18:00. A visit inside the warranty is
free for faults the warranty covers; outside the warranty, a fixed call-out fee applies, plus
parts. Spare parts stay available for at least ten years after the last washer of this model is
made.


19. DISPOSAL

Do not put an old washer in household waste. Take it to a recycling centre or ask the shop that
delivers the new one to take it back. Cut the power cord off and remove the door lock, so that
children cannot shut themselves inside.


20. SPECIFICATIONS

Capacity 8 kg. Size 60 x 60 x 85 cm. Weight 68 kg. Highest spin 1,400 rpm. Power 2,100 W.
Water pressure 0.1 to 1 MPa. Energy class A. Noise 74 dB while spinning.


21. WARRANTY

The washer is covered for two years on parts and one year on labour from the date of purchase.
Keep the receipt. The warranty does not cover damage from transport with the transit bolts out,
from installation that does not follow this manual, or from use other than household laundry.
To make a claim, call AquaWave service with the model, the serial number and the date of
purchase. Repairs under the warranty are carried out at your home by an AquaWave technician;
parts that are replaced become the property of AquaWave. A repair does not extend the warranty
period. Your legal rights as a consumer are not affected by this warranty.
"""


RECIPES = [
    ("Grandma's banana bread", "Makes one loaf.",
     ["3 very ripe bananas", "80 g melted butter", "150 g brown sugar", "1 egg, beaten", "1 tsp vanilla",
      "1 tsp baking soda", "a pinch of salt", "190 g plain flour", "a handful of chopped walnuts"],
     ["Heat the oven and grease a 23 x 13 cm loaf tin.", "Mash the bananas in a bowl and stir in the butter.",
      "Mix in the sugar, egg and vanilla, then the baking soda and salt.",
      "Fold in the flour and walnuts until just combined.",
      "Bake at 175 °C for 60 minutes, until a skewer comes out clean.",
      "Cool in the tin for 10 minutes, then on a rack."],
     "Freezes well in slices. Grandma used dates when there were no walnuts."),
    ("Red lentil soup", "Serves 6.",
     ["2 tbsp olive oil", "1 onion, chopped", "2 carrots, diced", "2 cloves garlic", "1 tsp cumin",
      "250 g red lentils", "1.5 litres vegetable stock", "1 lemon"],
     ["Soften the onion and carrots in the oil for 8 minutes.", "Add the garlic and cumin for one minute.",
      "Add the lentils and stock, bring to the boil and simmer for 25 minutes.",
      "Blend half of the soup and stir it back in.", "Season and finish with lemon juice."],
     "Thickens overnight; add water when reheating."),
    ("Weeknight chicken traybake", "Serves 4.",
     ["8 chicken thighs", "600 g small potatoes, halved", "2 red onions, in wedges", "1 lemon, quartered",
      "4 sprigs rosemary", "3 tbsp olive oil", "salt and pepper"],
     ["Toss everything in a large roasting tin.", "Roast at 200 °C for 45 minutes, turning once.",
      "Rest for 5 minutes and squeeze over the roasted lemon."],
     "Swap the potatoes for squash in autumn."),
    ("Overnight oats", "Serves 2.",
     ["100 g rolled oats", "250 ml milk", "100 g yoghurt", "1 tbsp chia seeds", "1 tbsp honey",
      "berries to serve"],
     ["Stir the oats, milk, yoghurt, chia and honey together.", "Cover and chill overnight.",
      "Top with berries in the morning."],
     "Keeps for two days in the fridge."),
    ("Mushroom risotto", "Serves 4.",
     ["30 g dried porcini", "1 litre hot stock", "1 onion", "300 g arborio rice", "125 ml white wine",
      "250 g chestnut mushrooms", "50 g parmesan", "30 g butter"],
     ["Soak the porcini in 250 ml hot water for 20 minutes.", "Soften the onion, add the rice and toast it.",
      "Add the wine, then the stock a ladle at a time for 18 minutes, stirring.",
      "Fry the fresh mushrooms and stir them in with the porcini and its liquid.",
      "Beat in the butter and parmesan and rest for 2 minutes."],
     "Leftovers make good arancini."),
    ("Black bean chilli", "Serves 6.",
     ["2 onions", "2 peppers", "3 cloves garlic", "2 tsp smoked paprika", "1 tsp chipotle",
      "2 tins black beans", "2 tins chopped tomatoes", "1 square dark chocolate"],
     ["Fry the onions and peppers until soft.", "Add the garlic and spices for a minute.",
      "Add the beans and tomatoes and simmer for 40 minutes.", "Stir in the chocolate and season."],
     "Better on the second day. Serve with rice or tortilla chips."),
    ("Lemon drizzle cake", "Makes one 20 cm cake.",
     ["225 g soft butter", "225 g caster sugar", "4 eggs", "225 g self-raising flour", "2 lemons",
      "85 g sugar for the drizzle"],
     ["Beat the butter and sugar until pale.", "Beat in the eggs one at a time, then fold in the flour and zest.",
      "Bake at 180 °C for 45 minutes.", "Mix the lemon juice with the drizzle sugar and pour over the warm cake."],
     "Prick the cake all over before the drizzle so it soaks in."),
    ("Shakshuka", "Serves 2.",
     ["1 onion", "1 red pepper", "2 cloves garlic", "1 tsp cumin", "1 tsp paprika", "1 tin tomatoes",
      "4 eggs", "feta and parsley"],
     ["Soften the onion and pepper.", "Add the garlic and spices, then the tomatoes; simmer for 10 minutes.",
      "Make four wells, crack in the eggs, cover and cook for 6 minutes.", "Finish with feta and parsley."],
     "Serve with crusty bread."),
    ("Pancakes", "Makes 12.",
     ["200 g plain flour", "2 tsp baking powder", "1 tbsp sugar", "a pinch of salt", "300 ml milk",
      "1 egg", "25 g melted butter"],
     ["Whisk the dry ingredients.", "Whisk in the milk, egg and butter until smooth.",
      "Cook ladles of batter in a hot pan until bubbles form, then flip."],
     "Sunday breakfast. Keep warm in a low oven."),
    ("Roast vegetable lasagne", "Serves 6.",
     ["2 courgettes", "1 aubergine", "2 peppers", "500 g passata", "12 lasagne sheets", "500 g ricotta",
      "125 g mozzarella", "basil"],
     ["Roast the chopped vegetables for 30 minutes.", "Stir the vegetables into the passata.",
      "Layer sauce, pasta and ricotta three times; top with mozzarella.",
      "Bake at 190 °C for 40 minutes and rest for 10 minutes."],
     "Assemble the day before and keep it in the fridge."),
    ("Thai green curry", "Serves 4.",
     ["2 tbsp green curry paste", "400 ml coconut milk", "500 g chicken or tofu", "1 aubergine",
      "a handful of green beans", "1 tbsp fish sauce", "1 lime", "Thai basil"],
     ["Fry the paste for one minute.", "Add the coconut milk and bring to a simmer.",
      "Add the chicken or tofu and vegetables and cook for 12 minutes.",
      "Season with fish sauce and lime and add the basil."],
     "Use half the paste for the children."),
    ("Apple crumble", "Serves 6.",
     ["6 cooking apples", "50 g sugar", "1 tsp cinnamon", "150 g flour", "100 g cold butter",
      "75 g demerara sugar", "50 g oats"],
     ["Slice the apples into a dish with the sugar and cinnamon.",
      "Rub the butter into the flour, then stir in the demerara and oats.",
      "Scatter over the apples and bake at 180 °C for 35 minutes."],
     "Custard, not cream."),
    ("Pesto pasta salad", "Serves 4.",
     ["300 g fusilli", "4 tbsp pesto", "200 g cherry tomatoes", "a handful of rocket", "50 g pine nuts",
      "1 ball mozzarella"],
     ["Cook the pasta and cool it under cold water.", "Toast the pine nuts in a dry pan.",
      "Toss everything together and season."],
     "Picnics and packed lunches."),
    ("Beef stew", "Serves 6.",
     ["1 kg stewing beef", "2 tbsp flour", "2 onions", "3 carrots", "2 sticks celery",
      "500 ml stock", "250 ml red wine", "2 bay leaves"],
     ["Toss the beef in the flour and brown it in batches.", "Soften the vegetables in the same pot.",
      "Add the beef, stock, wine and bay leaves.",
      "Cover and cook in the oven at 160 °C for 3 hours."],
     "Add dumplings for the last 25 minutes."),
    ("Granola", "Makes one large jar.",
     ["300 g oats", "100 g mixed nuts", "50 g seeds", "4 tbsp maple syrup", "3 tbsp oil",
      "1 tsp cinnamon", "100 g dried fruit"],
     ["Mix everything except the fruit.", "Spread on two trays and bake at 150 °C for 25 minutes, stirring "
      "twice.", "Cool and stir in the fruit."],
     "Keeps for a month in an airtight jar."),
    ("Carrot cake", "Makes one 20 cm cake.",
     ["200 g grated carrot", "175 g brown sugar", "3 eggs", "150 ml oil", "200 g self-raising flour",
      "1 tsp mixed spice", "100 g raisins", "cream cheese icing"],
     ["Whisk the sugar, eggs and oil.", "Fold in the flour, spice, carrot and raisins.",
      "Bake at 170 °C for 50 minutes.", "Ice when completely cold."],
     "Sam's birthday cake every year."),
]


# Blended soups share one method: (name, main vegetable, spice, simmer minutes, garnish, from).
SOUPS = [
    ("Carrot and ginger", "800 g carrots", "a thumb of fresh ginger", 20, "coriander", "Aunt Ruth"),
    ("Roast tomato", "1 kg tomatoes, roasted first", "1 tsp smoked paprika", 15, "basil oil", "the allotment"),
    ("Butternut squash", "1 butternut squash", "1 tsp ground cumin", 25, "toasted seeds", "Sam"),
    ("Pea and mint", "600 g frozen peas", "a handful of mint", 8, "a spoon of yoghurt", "Grandma"),
    ("Leek and potato", "3 leeks and 2 potatoes", "a bay leaf", 25, "chives", "Grandma"),
    ("Broccoli and stilton", "2 heads of broccoli", "a grating of nutmeg", 12, "crumbled stilton", "Ben"),
    ("Sweet potato and coconut", "3 sweet potatoes", "1 tbsp red curry paste", 20, "coconut cream",
     "Aiko"),
    ("Cauliflower cheese", "1 cauliflower", "1 tsp mustard", 15, "grated cheddar", "a cookbook"),
    ("Spiced parsnip", "6 parsnips", "1 tsp garam masala", 25, "crispy onions", "Grandma"),
    ("Celeriac and apple", "1 celeriac and 1 apple", "a sprig of thyme", 25, "walnuts", "Nora"),
    ("Watercress", "2 bags of watercress and 1 potato", "a pinch of nutmeg", 10, "crème fraîche",
     "a magazine"),
    ("Mushroom", "500 g mushrooms", "a sprig of thyme", 15, "truffle oil", "Priya"),
    ("Beetroot", "6 cooked beetroot", "1 tsp caraway", 15, "soured cream and dill", "Tomasz"),
    ("Courgette and pea", "4 courgettes and 200 g peas", "lemon zest", 10, "parmesan", "the allotment"),
    ("Red pepper and lentil", "4 red peppers and 100 g red lentils", "1 tsp harissa", 25, "feta", "Fatima"),
    ("Spinach and nutmeg", "500 g spinach and 1 potato", "plenty of nutmeg", 10, "a swirl of cream",
     "Grandma"),
    ("Jerusalem artichoke", "600 g Jerusalem artichokes", "a sprig of rosemary", 20, "hazelnuts",
     "a restaurant in Weston"),
    ("Tomato and red lentil", "2 tins tomatoes and 150 g red lentils", "1 tsp turmeric", 25,
     "lime and coriander", "Mei"),
    ("Chestnut and mushroom", "200 g cooked chestnuts and 300 g mushrooms", "a sprig of sage", 15,
     "a drizzle of cream", "Christmas 2023"),
    ("Corn chowder", "4 cobs of corn and 2 potatoes", "a pinch of chilli", 20, "crispy bacon",
     "Liam"),
    ("Fennel and orange", "2 fennel bulbs", "orange zest", 20, "fennel fronds", "a cookbook"),
    ("Green lentil and kale", "150 g green lentils and a bag of kale", "1 tsp ground coriander", 30,
     "a squeeze of lemon", "Grace"),
]
MENUS = [
    ("Summer barbecue for 12", "Burgers and halloumi, three salads (pesto pasta, potato, green), "
     "corn on the cob, and lemon drizzle cake. Buy two bags of charcoal and ice."),
    ("Christmas Eve for 8", "Chestnut and mushroom soup, then roast vegetable lasagne and a green "
     "salad with the house vinaigrette, then apple crumble and custard."),
    ("Birthday tea for 10", "Sandwiches, sausage rolls, crudités with the yoghurt and garlic dip, "
     "carrot cake and a fruit platter."),
    ("Curry night for 6", "Thai green curry, black bean chilli for anyone who does not eat meat, "
     "rice, flatbreads, and mango with lime."),
    ("Brunch for 6", "Shakshuka in two pans, pancakes, granola with yoghurt and berries, coffee and "
     "fresh orange juice."),
    ("Soup and bread lunch for 10", "Two soups (one blended, one chunky), a big loaf from the bakery, "
     "cheese, butter and pickles, and banana bread with tea. Keep the soups warm on the lowest heat "
     "and set out bowls, spoons and a ladle for each."),
    ("Picnic for 4", "Pesto pasta salad, a quiche from the shortcrust pastry, cherry tomatoes, "
     "crisps, strawberries and granola bars. Pack the dressing separately."),
]

DRESSINGS = [
    ("House vinaigrette", "3 tbsp olive oil, 1 tbsp red wine vinegar, 1 tsp Dijon mustard, a pinch of "
     "sugar"),
    ("Lemon and herb", "4 tbsp olive oil, juice of 1 lemon, a handful of chopped parsley and dill"),
    ("Honey mustard", "3 tbsp oil, 1 tbsp cider vinegar, 1 tbsp honey, 1 tbsp wholegrain mustard"),
    ("Tahini", "2 tbsp tahini, juice of half a lemon, 1 small garlic clove, water to thin"),
    ("Soy and ginger", "2 tbsp soy sauce, 1 tbsp rice vinegar, 1 tsp sesame oil, grated ginger"),
    ("Yoghurt and garlic", "100 g yoghurt, 1 crushed garlic clove, 1 tbsp olive oil, salt"),
    ("Balsamic", "3 tbsp olive oil, 1 tbsp balsamic vinegar, a pinch of salt"),
    ("Green goddess", "100 g yoghurt, 1 avocado, a bunch of basil and chives, lemon juice"),
]


BASICS = [
    ("Chicken stock", "Bones from a roast chicken, 1 onion, 1 carrot, 1 celery stick, a bay leaf, peppercorns.",
     "Cover with 2 litres of cold water, bring to a simmer, skim, and simmer gently for 3 hours. "
     "Strain and cool quickly.", "4 days in the fridge, 3 months frozen."),
    ("Vegetable stock", "2 onions, 2 carrots, 2 celery sticks, mushroom stalks, parsley stalks, a bay leaf.",
     "Cover with 2 litres of water and simmer for 45 minutes. Strain, pressing the vegetables.",
     "4 days in the fridge, 3 months frozen."),
    ("Basic tomato sauce", "2 tins tomatoes, 3 tbsp olive oil, 2 garlic cloves, a pinch of sugar, basil.",
     "Warm the sliced garlic in the oil, add the tomatoes and sugar, and simmer for 25 minutes. "
     "Add the basil at the end.", "5 days in the fridge, 3 months frozen."),
    ("White sauce", "40 g butter, 40 g plain flour, 500 ml milk, nutmeg.",
     "Melt the butter, stir in the flour for a minute, then whisk in the milk a little at a time. "
     "Simmer for 5 minutes, stirring, and season with nutmeg.", "2 days in the fridge."),
    ("Gravy", "Roasting juices, 1 tbsp flour, 400 ml stock, a splash of wine.",
     "Pour off most of the fat, stir the flour into the juices in the tin, add the wine and stock "
     "and simmer, scraping the tin, for 5 minutes. Strain.", "2 days in the fridge."),
    ("Shortcrust pastry", "250 g plain flour, 125 g cold butter, a pinch of salt, 3 to 4 tbsp cold water.",
     "Rub the butter into the flour, add water until it comes together, and chill for 30 minutes "
     "before rolling.", "2 days in the fridge, 2 months frozen."),
    ("Pizza dough", "500 g strong flour, 7 g dried yeast, 1 tsp salt, 325 ml warm water, 2 tbsp oil.",
     "Mix, knead for 10 minutes, and leave to rise for an hour. Makes four thin pizzas; bake on the "
     "hottest oven setting for 8 to 10 minutes.", "1 day in the fridge, 1 month frozen."),
    ("Custard", "500 ml milk, 4 egg yolks, 50 g sugar, 1 tbsp cornflour, vanilla.",
     "Heat the milk. Whisk the yolks, sugar and cornflour, pour on the hot milk, return to the pan "
     "and stir over a low heat until thick.", "2 days in the fridge."),
]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PLANS = [
    ["Red lentil soup and bread", "Weeknight chicken traybake", "Pesto pasta salad", "Shakshuka",
     "Pizza night", "Black bean chilli", "Roast chicken, then stock from the bones"],
    ["Leftover chilli with rice", "Thai green curry", "Carrot and ginger soup", "Mushroom risotto",
     "Fish and chips from the shop", "Roast vegetable lasagne", "Beef stew with dumplings"],
    ["Lasagne leftovers", "Pancakes for dinner", "Butternut squash soup", "Chicken traybake",
     "Pizza night", "Curry with friends", "Roast lamb"],
    ["Pea and mint soup", "Stew leftovers", "Pesto pasta salad", "Shakshuka", "Takeaway",
     "Barbecue if it is dry", "Sunday roast at Mum and Dad's"],
]


def recipes():
    out = ["# Family Recipe Collection",
           "Recipes collected from Grandma's notebook, friends and a few websites, typed up in 2025.\n"
           "Oven temperatures are for a conventional oven; take 20 degrees off for a fan oven."]
    for name, makes, ingredients, method, note in RECIPES:
        steps = "\n".join(f"{i}. {s}" for i, s in enumerate(method, 1))
        out.append(f"## {name}\n\n{makes}\n\nIngredients:\n"
                   + "\n".join(f"- {x}" for x in ingredients)
                   + f"\n\nMethod:\n{steps}\n\nNote: {note}")
    out.append("## Blended soups\n\nEvery soup below serves 4 and uses the same method: soften 1 "
               "chopped onion in 2 tbsp oil for 8 minutes, add the vegetables and the spice for 3 "
               "minutes, add 1.2 litres of stock, simmer until soft, blend and season.")
    for name, veg, spice, minutes, garnish, source in SOUPS:
        out.append(f"### {name} soup\n\nVegetables: {veg}. Spice: {spice}. Simmer for {minutes} "
                   f"minutes. Serve with {garnish}.\nFrom {source}. Freezes well for up to three "
                   "months; leave out the garnish until serving. For a thicker soup use less stock, "
                   "and for a richer one stir in a spoon of cream or a knob of butter at the end.")
    out.append("## Salad dressings\n\nShake everything in a jar. Each makes enough for a salad for "
               "four and keeps for a week in the fridge.\n\n"
               + "\n".join(f"- {name}: {what}." for name, what in DRESSINGS))
    for name, what, how, keeps in BASICS:
        out.append(f"### {name}\n\n{what}\n\n{how}\n\nKeeps: {keeps}")
    for week, days in enumerate(PLANS, 1):
        out.append(f"## Meal plan, week {week}\n\n"
                   + "\n".join(f"- {day}: {meal}" for day, meal in zip(WEEKDAYS, days)))
    out.append("## Menus for a crowd\n\nScale the recipes by the number of guests, but not the salt or "
               "the spices: add those to taste. Cook what you can the day before.\n\n"
               + "\n\n".join(f"**{name}.** {menu}" for name, menu in MENUS))
    out.append("## Pantry list\n\nAlways in the cupboard: plain and self-raising flour, caster and "
               "brown sugar, rolled oats, red lentils, arborio and basmati rice, dried pasta, tinned "
               "tomatoes, tinned black beans and chickpeas, coconut milk, stock cubes, olive oil, "
               "vegetable oil, red wine and cider vinegar, soy sauce, Dijon and wholegrain mustard, "
               "honey, maple syrup, cumin, smoked paprika, cinnamon, mixed spice, chilli flakes, bay "
               "leaves, baking powder, baking soda and vanilla.\n\nAlways in the fridge: butter, eggs, "
               "milk, yoghurt, parmesan, cheddar, lemons, garlic, ginger and a bag of spinach.")
    table = "\n".join(f"| {c} °C | {round(c * 9 / 5 + 32)} °F |" for c in range(140, 241, 10))
    out.append("## Kitchen notes\n\nOven temperatures:\n\n| Celsius | Fahrenheit |\n|---|---|\n" + table
               + "\n\nOne cup of flour is about 125 g, one cup of sugar about 200 g, one cup of butter "
               "about 225 g. A tablespoon is 15 ml and a teaspoon 5 ml.")
    return "\n\n".join(out) + "\n"


MONTHS = [
    ("14 January 2026", "Annual dues remain $600 for 2026, payable by 31 March.",
     "The boiler in building B was serviced; the gutters on the north side need clearing.",
     "Winter clean-up day set for 7 February.",
     "Residents asked for more lighting on the path to the bicycle shed."),
    ("11 February 2026", "Dues received so far: 41 of 60 units.",
     "Snow clearing contract renewed with Pike Grounds for one more winter.",
     "Twelve residents came to the clean-up day and filled two skips.",
     "A resident asked whether the laundry room can stay open until 23:00."),
    ("11 March 2026", "Dues received: 57 of 60 units; reminders sent to the other three.",
     "Laundry room hours extended to 23:00 as a trial.",
     "Plant swap planned for April in the courtyard.",
     "Concern about cars parked in the fire lane on weekends."),
    ("8 April 2026", "All dues received. Reserve fund at $48,300.",
     "Two quotes received for repainting the hallways.",
     "The plant swap brought about 30 people.",
     "Request for a second bicycle rack near building C."),
    ("13 May 2026", "Pool maintenance invoice paid: $2,150.",
     "The pool reopens on 1 June after the new pump was fitted.",
     "Summer barbecue booked for 20 June.",
     "Residents asked for clearer pool rules for guests."),
    ("10 June 2026", "Insurance for the common areas renewed at $6,900 a year.",
     "Hallway repainting starts in building A on 6 July.",
     "Pool rules posted: two guests per unit, no glass.",
     "A resident offered to run a book exchange shelf in the lobby."),
    ("8 July 2026", "Painting deposit paid: $4,000.",
     "Painting of building A is on schedule; building B follows in August.",
     "The summer barbecue raised $340 for the garden fund.",
     "Noise from late-night pool use was raised."),
    ("12 August 2026", "Painting second payment: $4,000.",
     "The elevator in building C failed twice; the contractor replaced a door sensor.",
     "Back-to-school bake sale on 5 September.",
     "Request to fix the broken bench by the playground."),
    ("9 September 2026", "The board voted to raise annual dues to $640 starting January 2027.",
     "Roof inspection booked for 2 October for all three buildings.",
     "The bake sale raised $210.",
     "Several residents asked why the dues are going up; the treasurer pointed to insurance and the "
     "roof."),
    ("14 October 2026", "Budget for 2027 presented and accepted.",
     "The roof inspection found loose flashing on building B; repair scheduled for November.",
     "Halloween decorations in the lobby from 25 October.",
     "A resident asked for a car-sharing parking space."),
]
LETTERS = [
    "A letter from the city about the new recycling bins, which arrive in March.",
    "The insurer asked for the fire alarm test certificates; the secretary sent them.",
    "A thank-you card from the family in 2C for the help after their flood.",
    "The painting firm sent its method statement and the dates it needs access.",
    "The pool company confirmed the safety inspection for 28 May.",
    "A complaint from a neighbouring building about the hedge on the boundary.",
    "The city confirmed that the street will be resurfaced in September.",
    "The elevator contractor proposed a new maintenance contract at $3,600 a year.",
    "Two residents wrote in support of a car-sharing space.",
    "The roofer's inspection report arrived on 6 October and was shared with all units.",
]
GARDEN = [
    "The garden committee will prune the fruit trees in the courtyard in February.",
    "Bulbs planted in the autumn are coming up along the front path.",
    "Compost bins will be emptied onto the shared beds before the plant swap.",
    "The herb bed is open for everyone to pick from; please cut, do not pull.",
    "Watering rota for the summer is on the noticeboard; nine volunteers so far.",
    "The new bench by the herb bed has been ordered with money from the garden fund.",
    "The lawn was left long in parts to help the bees, as agreed in May.",
    "The courtyard tap leaked and was repaired by the caretaker.",
    "Leaf clearing day set for 31 October.",
    "Spring bulbs bought for the front beds, to be planted in November.",
]
SAFETY = [
    "The fire alarm test on 6 January passed in all three buildings.",
    "Two emergency lights in building C were replaced.",
    "A reminder that bicycles must not be left in the stairwells; three were tagged.",
    "The front door of building B was not closing; the closer was adjusted.",
    "Pool gate code changes on 1 June; the new code goes to each unit by letter.",
    "Smoke alarms in the common areas were tested; one battery was replaced.",
    "A ladder was left in the courtyard overnight by the painters; they have been told.",
    "Children were seen playing in the car park; parents are asked to keep them to the playground.",
    "Scaffolding for the roof inspection will stand for two days; keep clear of it.",
    "The clocks change on 25 October; the timers for the outside lights were reset.",
]
NEXT = ["11 February", "11 March", "8 April", "13 May", "10 June", "8 July", "12 August",
        "9 September", "14 October", "11 November"]
PEOPLE = ["Mei Tan (chair)", "Oscar Ruiz (treasurer)", "Fatima Haddad (secretary)", "Jordan Avery",
          "Liam O'Neill", "Grace Mensah", "Tomasz Nowak", "Aiko Sato", "Ben Carter", "Nora Lind"]


def meeting_notes():
    out = ["# Alder Street Residents' Association: meeting notes, 2026",
           "Meetings are held on the second Wednesday of each month at 19:00 in the community room of "
           "building A. Notes are taken by the secretary and approved at the next meeting."]
    for i, (date, money, upkeep, event, floor) in enumerate(MONTHS):
        present = ", ".join(PEOPLE[:3] + PEOPLE[3 + i % 4: 6 + i % 4])
        balance = 18000 + 1350 * i
        out.append(f"""## Meeting of {date}

Present: {present}. Apologies from {PEOPLE[9 - i % 3]}.

The notes of the previous meeting were approved without changes.

### Treasurer's report

{money} The operating account stands at ${balance:,}. Spending this month was within the
budget for cleaning, gardening and electricity in the common areas. The treasurer will circulate
the full statement by email. Receipts for anything a volunteer bought for the association go to
the treasurer within a month, with the date and what it was for, so they can be paid back.

### Maintenance

{upkeep} The caretaker's report was noted. Residents are reminded to report repairs through the
association's form rather than by message to board members.

### Events

{event} Volunteers are welcome; sign up on the lobby noticeboard.

### Correspondence

{LETTERS[i]} Letters to the association go in the post box in the lobby of building A or to
the association's email address; the secretary answers within two weeks.

### Garden

{GARDEN[i]} Anyone who wants to help in the garden can come on the first Saturday of the month
from 10:00; tools and gloves are in the shed by the bicycle rack.

### Safety

{SAFETY[i]} Report anything unsafe in the common areas to the caretaker the same day; for an
emergency call the emergency services first.

### Open floor

{floor} The chair thanked everyone for coming.

### Next meeting

The next meeting is on {NEXT[i]} 2026 at 19:00 in the community room of building A. Send agenda
items to the secretary at least a week before.

### Action items

- Secretary to post these notes on the noticeboard within a week.
- Treasurer to send the monthly statement.
- Chair to follow up the open-floor items before the next meeting.""")
    return "\n\n".join(out) + "\n"


def status(date, done, countertop, spent, risks):
    return f"""# Kitchen renovation: status update, {date}

Contractor: Birch & Sons Renovations. Project lead: Callum Birch. Budget: $25,000.

## Done since the last update

{done}

## Schedule

{countertop}
Plumbing and the sink are connected the day after the countertop is in. Painting and the
backsplash follow the week after.

## Budget

Spent to date: ${spent:,} of the $25,000 budget. The contingency of $2,000 is still untouched.

## Risks and decisions

{risks}

## Contacts

Callum Birch, 555-0131. Site hours are 8:00 to 16:30, Monday to Friday; the site is locked at
weekends. Photos of the work so far are in the shared album "Kitchen 2026". Invoices go to the
email address on the contract and are paid within 14 days.

## Next update

In about four weeks, or sooner if the schedule changes.
"""


DOCS = {
    "lease-alder-street.md": lease_alder,
    "lease-maple-avenue-2024.md": lease_maple,
    "trip-plan-lisbon.md": trip_lisbon,
    "washer-manual.txt": washer_manual,
    "recipes.md": recipes,
    "residents-meeting-notes-2026.md": meeting_notes,
    "project-status-2026-10.md": lambda: status(
        "9 October 2026",
        "The old units were removed and the walls were patched. New wiring for the island was "
        "inspected and passed. The cabinets were installed on 2 October and the doors adjusted.",
        "Countertop installation is now scheduled for 3 November 2026, after the quartz slab was "
        "re-cut to fix a measuring error at the fabricator.",
        18400,
        "The re-cut moved the countertop by three weeks; the fabricator pays for the re-cut. We "
        "chose the brushed brass handles. The dishwasher stays the old one until the spring."),
    "project-status-2026-08.md": lambda: status(
        "14 August 2026",
        "The design was signed off and the cabinets ordered. The quartz slab was chosen at the "
        "stone yard. The old kitchen will be removed in September.",
        "Countertop installation is scheduled for 14 October 2026, two weeks after the cabinets.",
        9200,
        "Lead times for cabinets are six weeks. We may need a temporary kitchen in the dining "
        "room for about a month."),
    "bank-letter-savings-rate.txt": lambda: """Northgate Savings Bank
12 Market Square, Riverton

Jordan Avery
42 Alder Street, Apartment 3B
Riverton

2 October 2026

Your Everyday Saver account ending 4821

Dear Jordan Avery,

We are writing to tell you about a change to the interest rate on your Everyday Saver account.

From 1 November 2026 the interest rate on your Everyday Saver account will change from 3.10% APY
to 2.75% APY. Interest is still calculated daily and paid monthly.

Why we are making this change: the central bank lowered its base rate in September, and the
rates we pay on savings follow it.

You do not need to do anything. If you would like to move your money, you can close the account
or move to another of our accounts at any time without a fee. Our Fixed Saver 12 pays 3.40% APY
for balances you can leave untouched for a year.

Your balance on 30 September 2026 was $12,480.55. Your deposits with us are protected by the
national deposit insurance scheme up to $250,000 per person.

If you have questions, call us on 555-0199, Monday to Saturday, 8:00 to 20:00, or visit any
branch.

Yours sincerely,

Helen Marsh
Head of Savings, Northgate Savings Bank
""",
    "insurance-renters-renewal.txt": lambda: """Cedar Mutual Insurance
Renters insurance: renewal notice

Policyholder: Jordan Avery
Insured address: 42 Alder Street, Apartment 3B, Riverton
Policy number: RI-208815
Renewal period: 1 November 2026 to 31 October 2027

Thank you for staying with Cedar Mutual. Your policy renews automatically on 1 November 2026
with the cover below. If anything has changed, for example you bought valuable items or a
roommate moved in, please tell us before the renewal date.

YOUR COVER
Personal property: $30,000, replacement cost
Liability: $100,000 per occurrence
Medical payments to others: $2,000 per person
Additional living expenses: up to 12 months
Deductible: $500 per claim.

YOUR PRICE
Your premium: $18.50 per month.
The premium is collected by direct debit on the 1st of each month. Paying for the whole year at
once costs $210 instead.

WHAT IS NOT COVERED
Floods and earthquakes, wear and tear, pests, and property of people who are not named on the
policy. Bicycles over $1,500 and jewellery over $2,500 must be listed separately.

HOW TO CLAIM
Call 555-0170 at any time or claim online. Keep receipts and photos of valuable items.
""",
    "insurance-auto-card.txt": lambda: """Cedar Mutual Insurance
Auto insurance card and summary

Named driver: Jordan Avery
Policy number: AU-55102
Vehicle: 2019 Corvo Hatchback, grey, plate on file
Policy period: 15 March 2026 to 14 March 2027

COVER AND DEDUCTIBLES
Liability: 100/300/100
Collision deductible: $1,000.
Comprehensive deductible: $250.
Roadside assistance: included, call 555-0171.
Rental car: up to $40 a day for 30 days after a covered claim.

PREMIUM
$96 per month by direct debit.

WHO MAY DRIVE
The named driver, and anyone over 25 with the named driver's permission and a full licence.
Drivers under 25 must be added to the policy before they drive.

DISCOUNTS ON THIS POLICY
Safe driver (no claims for five years), multi-policy (renters insurance with Cedar Mutual) and
paperless billing.

Your no-claims record: 6 years. A claim that was not your fault does not affect it. Glass
repairs are covered without a deductible when you use one of our approved repairers.

Keep this card in the vehicle. Show it to the police after an accident or when asked.
In an accident: make sure everyone is safe, call the police if anyone is hurt, take photos,
exchange details, and report the claim within 72 hours.
""",
    "dentist-appointment.txt": lambda: """Bright Smile Dental
48 Elm Row, Riverton. Phone 555-0123.

Appointment reminder

Patient: Jordan Avery

Your next appointment is on Tuesday 3 November 2026 at 08:30 with Dr. Imani Okafor.
Visit: six-monthly check-up and cleaning with the hygienist. Allow 45 minutes.

Please bring your insurance card and a list of any medicines you take. If you need to change
the time, call us at least 24 hours before, or a missed-appointment fee of $40 applies.

Parking is free behind the building. The practice is on the first floor; there is a lift.

Before your visit: brush and floss as usual. If you have had any changes to your health, a new
medicine or a pregnancy since your last visit, tell the receptionist when you arrive.

Your plan: Bright Smile Care, $22 a month, covers two check-ups and two cleanings a year and 20%
off other treatment. Your insurer pays the rest of the cleaning fee directly to us.

Your recent visits:
- 4 May 2026: check-up and cleaning. No new cavities. X-rays taken.
- 10 November 2025: small filling, lower left molar.
- 2 May 2025: check-up and cleaning.
""",
    "car-service-record.md": lambda: """# Car service record: 2019 Corvo Hatchback

| Date | Mileage | Work | Garage | Cost |
|---|---|---|---|---|
| 12 April 2024 | 31,200 | Oil and filter, tyre rotation | Quayside Motors | $89 |
| 2 October 2024 | 35,900 | Front brake pads | Quayside Motors | $240 |
| 18 March 2025 | 39,400 | Annual service, air filter, wipers | Quayside Motors | $310 |
| 9 September 2025 | 42,100 | Oil and filter, battery test | Quayside Motors | $95 |
| 20 March 2026 | 44,800 | Annual service, cabin filter, two new tyres | Quayside Motors | $520 |

Next service due at 48,000 miles or by 15 March 2027, whichever comes first.

## Notes

- Tyre pressure: 2.4 bar front and back, 2.6 bar when loaded.
- The check-engine light came on once in July 2025; it was a loose fuel cap.
- The spare wheel is a space saver; do not drive over 80 km/h on it.
- Winter tyres are stored at Quayside Motors, ticket number 1187.
- Quayside Motors: 3 Dock Road, 555-0177, open 7:30 to 17:30 on weekdays.
- The registration and the roadworthiness certificate are in the glove box. The certificate
  must be renewed with the annual service in March.
""",
    "employer-benefits-summary.md": lambda: """# Benefits summary for 2027: Lattice Point Software

This summary is for full-time employees. The plan documents decide in case of any difference.

## Open enrollment

Open enrollment for 2027 benefits ends on 15 November 2026. Make your choices in the benefits
portal. If you do nothing, your 2026 medical and dental choices carry over, but your flexible
spending account does not.

## Medical

Three plans: Bronze (high deductible, with a health savings account), Silver and Gold. The
company pays 85% of the premium for employees and 70% for dependants.

## Dental and vision

Dental covers two cleanings a year at 100%. Vision covers one eye exam a year and $200 for
glasses or contact lenses every two years.

## Retirement

The company matches 100% of your contributions to the 401(k) up to 4% of salary. Matching
contributions vest after one year.

## Time off

20 days of paid time off a year, plus 10 public holidays and the week between Christmas and New
Year. Up to 5 unused days carry over.

## Other

Parental leave of 16 weeks at full pay. A $600 yearly wellness allowance for gyms, classes or
equipment. Commuter benefits up to the legal limit.

Questions: benefits@lattice-point.example or the HR desk on the third floor.
""",
    "gym-membership.txt": lambda: """Riverside Fitness Club: membership agreement

Member: Jordan Avery
Membership: Standard, all clubs, start date 10 January 2026

Monthly fee: $42, charged on the 10th of each month.
Joining fee: $25, paid once.
Classes: included, book in the app up to 7 days ahead.
Guest passes: two a month.

Freezing: you can freeze the membership for one to three months a year for $5 a month.

Cancelling: To cancel, give 30 days' written notice at the front desk or by email to
members@riverside-fitness.example. The last fee is charged within the notice period.

Opening hours: 6:00 to 22:00 on weekdays, 8:00 to 20:00 at weekends.
Please bring a towel and use the lockers; the club is not responsible for valuables.

Club rules: wipe down equipment after use, re-rack weights, no outdoor shoes in the studios, and
book classes you can attend: three missed classes in a month pause class booking for a week.
Members under 16 may use the gym only with a parent and after an induction.

Personal training: sessions cost $35 each or $160 for five. Your membership includes one free
session in the first month; it was used on 17 January 2026.
""",
    "phone-plan-bill-2026-09.txt": lambda: """Brightwave Mobile
Monthly bill, September 2026

Account holder: Jordan Avery
Account number: 7720-1185-03
Billing period: 1 to 30 September 2026

Plan: Unlimited Talk 20 GB, $55.00 per month.

Usage this month
Calls: 312 minutes (unlimited)
Texts: 148 (unlimited)
Data: 14.2 GB of 20 GB
Roaming: none

Charges
Plan                          $55.00
Device instalment (11 of 24)  $18.75
Taxes and fees                 $6.12
Total due                     $79.87

The total will be collected by direct debit on 15 October 2026.
Travelling abroad? A roaming day pass costs $10 per day in 60 countries.

Your device: Pixelon 8, instalment 11 of 24. You can upgrade after instalment 18.

News from Brightwave: from January, plans with 20 GB or more include 5 GB of roaming data in
Europe at no extra cost. Your plan qualifies, so you do not need to do anything.

Questions about this bill: call 555-0140 or chat in the Brightwave app. To dispute a charge,
tell us within 30 days of the bill date.

Previous bills: $79.87 (August), $79.87 (July), $89.87 (June, with one roaming day pass).
""",
    "book-club-notes.md": lambda: """# Book club

We meet once a month, rotating between homes. Bring a snack or a drink.

## October meeting (8 October, at Sam's)

We discussed *The Glass Orchard*. Most of us liked the first half more than the ending. Priya
thought the narrator was unreliable from the start; Ben did not. Average rating 3.5 of 5.

## Next meeting

Next meeting: Thursday 12 November at Priya's place, 7 pm.
Book: *Salt and Weather* by Ines Varga, about a lighthouse keeper's family over three
generations. It is about 320 pages, so start early.

## Ideas for next year

- A classic every quarter.
- One meeting at the library's author talk.
- A summer meeting in the park.
- Ben suggested a short-story month in December, when everyone is busy.

## Read this year

- January: *The Long Field*, at Ben's.
- March: *Notes from a Small Harbour*, at Aiko's.
- May: *The Cartographer's Daughter*, at Jordan's.
- July: *Rain on Tin*, in the park.
- September: *Seven Kitchens*, at Nora's.

## Members

Sam, Priya, Ben, Aiko, Jordan, Nora. Grace may join in the new year.
""",
    "garden-planting-log.md": lambda: """# Garden planting log, 2026

Allotment plot 22, Riverton community garden.

| Date | What | Where | Notes |
|---|---|---|---|
| 21 March | Broad beans | Bed 1 | Sown direct, two rows |
| 4 April | Early potatoes | Bed 2 | Chitted since February |
| 18 April | Lettuce, radish | Bed 3 | Under fleece |
| 2 May | Tomatoes (6 plants) | Greenhouse | Gardener's Delight and San Marzano |
| 16 May | Courgettes, squash | Bed 4 | Lots of compost |
| 30 May | Runner beans | Bed 1 | After the broad beans |
| 12 July | Leeks | Bed 2 | After the potatoes |
| 10 October | Garlic | Bed 3 | Twelve cloves, hardneck |

## Notes

The water butt was empty for most of July; a second one would help. Slugs took most of the first
lettuce. The tomatoes did well in the greenhouse but split after heavy watering in August.
The plot fee of $45 for 2027 is due in January.

## Plans for 2027

Rotate the beds: potatoes to bed 4, squash to bed 2. Try sweetcorn in bed 3 after the garlic.
Build a second compost bay and put netting over the brassicas before the pigeons find them.
""",
    "vet-records-miso.md": lambda: """# Miso: vet records

Miso is a female tabby cat, born about April 2021, adopted from the Riverton shelter in 2022.
Microchip number on the shelter certificate. Spayed.

Vet: Parkside Veterinary Clinic, 555-0188.

## Vaccinations

- Cat flu and enteritis: given 14 February 2026, next due 14 February 2027.
- Rabies booster due: 14 February 2027.

## Visits

- 14 February 2026: annual check and vaccinations. Healthy.
- 2 September 2026: limping on the front left paw. A small cut, cleaned; healed in a week.
  Weight on 2 September 2026: 4.6 kg.

## Food and care

Dry food morning and evening, wet food at night. Flea treatment monthly. Insurance: PetSure,
policy PS-30984.

## Notes

Weight history: 4.3 kg at adoption, 4.5 kg in February 2025, 4.4 kg in February 2026. The vet
says 4 to 5 kg is right for her size.

Miso hides when visitors come and needs a carrier with a towel over it for the vet. Worming
tablet every three months, hidden in a treat. She does not like being brushed but tolerates it
after dinner. When we travel, Sam feeds her and changes the litter.
""",
    "tax-documents-checklist.md": lambda: """# Tax documents checklist for the 2026 return

Gather these by the end of January 2027 and put them in the blue folder.

- [ ] Wage statement from Lattice Point Software
- [ ] Interest statement from Northgate Savings Bank
- [ ] 401(k) contribution summary
- [ ] Receipts for charitable donations (the shelter, the garden fund)
- [ ] Health savings account statement
- [ ] Records of any freelance work and its invoices
- [ ] Last year's return, for reference

## Notes

Ask the accountant whether the home office counts this year. Keep receipts for the kitchen
renovation, in case the energy-efficient appliances qualify for a credit.

Accountant: Ruth Feld, Feld and Partners, 555-0150. Book the appointment in December.

## Last year

The 2025 return was filed on 3 March 2026. The refund of $412 arrived on 21 March. The
accountant's fee was $275.

## Deadlines

The filing deadline is 15 April 2027. Estimated tax for any freelance income is due quarterly;
the accountant will say whether that applies. Keep every record for at least seven years, in
the blue folder for paper and the "Taxes" folder for scans.
""",
    "wedding-rsvp-and-gift.md": lambda: """# Cousin Leah's wedding

Leah Avery and Marco Bianchi are getting married on Saturday 5 December 2026 at 15:00, at the
Old Mill, Weston. Reception to follow in the same place.

Please RSVP by 1 November 2026 on the couple's website, with any dietary needs.

## Plan

- Hotel: two nights at the Weston Inn, booked, confirmation 55102.
- Drive down on Friday afternoon; about three hours.
- Dress code: formal, winter.

## Gift

The registry is at Homestead. We chose the cast-iron casserole dish ($180), shared with Mum and
Dad. Bring a card.

## On the day

- 14:30: arrive at the Old Mill; ushers show guests to the barn.
- 15:00: ceremony.
- 16:00: photos by the mill pond, drinks on the terrace.
- 18:00: dinner and speeches.
- 20:30: band and dancing until midnight. Taxis booked for 00:15.

If anything changes, ask Leah's sister Rosa, who is organising the day.

Children are welcome at the ceremony and the dinner; there is a quiet room with games. The Old
Mill has parking for 60 cars, and the Weston Inn is a ten-minute walk along the river path.
""",
    "home-router-setup.txt": lambda: """Home internet: router setup notes

Provider: Riverlink Fibre. Account number RF-448812. Support 555-0166.
Router: the white box in the hall cupboard.

Wi-Fi network name: Alder3B-Home
Wi-Fi password: maple-otter-58
Guest network: Alder3B-Guest, password on the fridge card

Router admin page: http://192.168.1.1 (user admin, password on the sticker under the router).

If the internet drops:
1. Check the light on the fibre box. Green is fine; red means a line fault, call support.
2. Restart the router: switch it off for 30 seconds and on again.
3. If one device cannot connect, forget the network on that device and join again.

Speed: 500 Mbps down, 100 Mbps up. Measured 470 down on 1 October 2026.
The printer and the smart speaker are on the 2.4 GHz band; everything else uses 5 GHz.

Devices on the network: two laptops, two phones, the printer in the study, the smart speaker in
the kitchen and the television. The router updates itself at 3:00 on Sundays.

Contract: 24 months from 1 September 2025 at $45 a month, then $60 a month unless renewed. The
router stays with Riverlink; return it in its box when the contract ends.
""",
    "cabin-weekend-plan.md": lambda: """# Pine Hollow cabin weekend

Pine Hollow cabin weekend, Friday 20 to Sunday 22 November 2026.

Address: 7 Larch Lane, Pine Hollow. About two and a half hours' drive.

## Getting in

Check-in from 16:00 on Friday, check-out by 11:00 on Sunday.
The key is in the lockbox by the side door; the lockbox code is 2719.
The host says there is Wi-Fi in the cabin; the details are on a card inside.

## Who is coming

Jordan, Sam, Aiko and Ben. Two cars. Aiko brings the board games.

## Food

Friday: chilli (make at home, freeze, bring). Saturday: pancakes, a packed lunch for the hike,
traybake for dinner. Sunday: leftovers and coffee.

## To bring

Hiking boots, rain jackets, firewood is provided, torches, the cooler bag, a speaker.

## Hike

The Ridge Loop, 11 km, starts from the car park at the end of Larch Lane.

## House rules from the host

No smoking inside. Quiet after 22:00. Put the rubbish in the bear-proof bin by the gate. Leave
the key in the lockbox when you go. The cost is $480 for the two nights, split four ways.
""",
    "utility-bill-electric-2026-09.txt": lambda: """Riverton Power
Electricity bill

Customer: Jordan Avery
Supply address: 42 Alder Street, Apartment 3B
Account: 3301-7754
Bill date: 28 September 2026

Usage, 26 August to 25 September 2026: 412 kWh
Energy charge: 412 kWh at $0.19 = $78.28
Standing charge: 30 days at $0.42 = $12.60
Taxes: $5.52
Amount due: $96.40 by 20 October 2026.

Your usage was 8% lower than the same month last year.
Pay online, by phone on 555-0110, or by direct debit.

Meter readings
Previous reading: 18,204 (25 August 2026, actual)
This reading: 18,616 (25 September 2026, actual)

Ways to save
Switch off standby at the wall, wash clothes at 30 °C, and use a lid on pans. A smart meter can
show what you use each hour; ask us to fit one at no cost.

Your tariff: Standard Variable. A fixed tariff for 12 months at $0.17 per kWh is available to
you from 1 November.

Power cut? Call 105 or check the outage map on our website. Never touch fallen power lines.

Your last three bills: $104.75 (June), $88.10 (July), $91.30 (August). All paid on time.
""",
}

# Files the skill must never read: a hidden note and a PDF. Each holds the answer to a question
# that no document answers, so reading either shows up in the abstain rate.
TRAPS = {
    ".draft-notes.md": "Cabin Wi-Fi password: pinecone-42 (from the host's email).\n",
    "scan-passport.pdf": "%PDF-1.4\n% scanned page, no text layer\nPassport number: X1234567\n%%EOF\n",
}


def fact(file, quote):
    return {"file": file, "quote": quote}


LEASE = "lease-alder-street.md"
RENT = fact(LEASE, "Monthly rent: $1,850, due on the first day of each month.")

QUESTIONS = [
    ("What is the monthly rent for the apartment at 42 Alder Street?", [RENT]),
    ("What is the late fee if the Alder Street rent is paid late?",
     [fact(LEASE, "If rent is not received by the fifth day of the month, the tenant pays a late fee of $75.")]),
    ("When is the kitchen countertop installation scheduled?",
     [fact("project-status-2026-10.md", "Countertop installation is now scheduled for 3 November 2026")]),
    ("What time does my flight to Lisbon depart?",
     [fact("trip-plan-lisbon.md", "Outbound flight GL 218 departs Newark at 18:40 on Thursday 22 October")]),
    ("What does error code E21 mean on the washer?",
     [fact("washer-manual.txt", "E21: The drum did not drain within 10 minutes.")]),
    ("How often should I clean the washer's drain pump filter?",
     [fact("washer-manual.txt", "Clean the drain pump filter every three months.")]),
    ("At what temperature and for how long do I bake the banana bread?",
     [fact("recipes.md", "Bake at 175 °C for 60 minutes, until a skewer comes out clean.")]),
    ("How much will the annual residents' association dues be from January 2027?",
     [fact("residents-meeting-notes-2026.md",
           "The board voted to raise annual dues to $640 starting January 2027.")]),
    ("What will the interest rate on my savings account change to?",
     [fact("bank-letter-savings-rate.txt", "will change from 3.10% APY\nto 2.75% APY")]),
    ("What is the deductible on my renters insurance?",
     [fact("insurance-renters-renewal.txt", "Deductible: $500 per claim.")]),
    ("When is my next dentist appointment?",
     [fact("dentist-appointment.txt",
           "Your next appointment is on Tuesday 3 November 2026 at 08:30 with Dr. Imani Okafor.")]),
    ("When is Miso's rabies booster due?",
     [fact("vet-records-miso.md", "Rabies booster due: 14 February 2027.")]),
    ("How much notice do I need to give to cancel my gym membership?",
     [fact("gym-membership.txt", "To cancel, give 30 days' written notice at the front desk or by email")]),
    ("What is the lockbox code for the cabin key?",
     [fact("cabin-weekend-plan.md", "the lockbox code is 2719")]),
    ("How much do I pay each month for the Alder Street rent and the renters insurance together?",
     [RENT, fact("insurance-renters-renewal.txt", "Your premium: $18.50 per month.")]),
    ("Is my dentist appointment before or after my return flight from Lisbon?",
     [fact("dentist-appointment.txt",
           "Your next appointment is on Tuesday 3 November 2026 at 08:30 with Dr. Imani Okafor."),
      fact("trip-plan-lisbon.md", "Return flight GL 219 leaves Lisbon at 10:15 on Thursday 29 October")]),
    ("Is the kitchen countertop installation before or after the wedding RSVP deadline?",
     [fact("project-status-2026-10.md", "Countertop installation is now scheduled for 3 November 2026"),
      fact("wedding-rsvp-and-gift.md", "Please RSVP by 1 November 2026")]),
    ("Can my cat Miso live with me at Alder Street, and how much does Miso weigh?",
     [fact(LEASE, "One cat is allowed with a pet deposit of $300."),
      fact("vet-records-miso.md", "Weight on 2 September 2026: 4.6 kg.")]),
    ("Which costs more each month, the gym membership or the phone plan?",
     [fact("gym-membership.txt", "Monthly fee: $42, charged on the 10th of each month."),
      fact("phone-plan-bill-2026-09.txt", "Plan: Unlimited Talk 20 GB, $55.00 per month.")]),
    ("Is the next book club meeting before the Pine Hollow cabin weekend?",
     [fact("book-club-notes.md", "Next meeting: Thursday 12 November at Priya's place, 7 pm."),
      fact("cabin-weekend-plan.md", "Pine Hollow cabin weekend, Friday 20 to Sunday 22 November 2026.")]),
    ("What is the Wi-Fi password at the Pine Hollow cabin?", []),
    ("What is my passport number?", []),
    ("What is the monthly rent for 17 Maple Avenue in 2027?", []),
    ("How long is the warranty on the dishwasher?", []),
]


def build(out):
    """Writes every file into `out` and returns their names."""
    files = {name: make() for name, make in DOCS.items()}
    files.update(TRAPS)

    def placed(f):
        """The fact with the line its quote starts on, which the recall score checks."""
        text = files[f["file"]]
        return {**f, "line": text[:text.index(f["quote"])].count("\n") + 1}

    questions = [{"id": f"q{i:02d}", "question": q, "facts": [placed(f) for f in facts]}
                 for i, (q, facts) in enumerate(QUESTIONS, 1)]
    files["questions.json"] = json.dumps(questions, indent=2, ensure_ascii=False) + "\n"
    for name, text in files.items():
        # open(), not write_text(newline=), which needs Python 3.10; macOS ships 3.9.
        with open(out / name, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    return sorted(files)


def problems():
    """What is wrong with the set this script writes, as sentences; empty when it is sound."""
    out = []
    texts = {name: make() for name, make in DOCS.items()}
    sizes = {name: len(text.encode()) for name, text in texts.items()}
    if len(texts) != 24:
        out.append(f"{len(texts)} documents, not 24")
    out += [f"{name}: {size} bytes, not 1 to 30 KB" for name, size in sizes.items()
            if not KB <= size <= 30 * KB]
    big = [name for name, size in sizes.items() if size > 20 * KB]
    if len(big) != 3:
        out.append(f"{len(big)} documents over 20 KB, not 3: {big}")
    mix = [len({f["file"] for f in facts}) for _, facts in QUESTIONS]
    if (mix.count(1), mix.count(2), mix.count(0), len(mix)) != (14, 6, 4, 24):
        out.append(f"question mix {mix}, not 14 one-document, 6 two-document, 4 empty")
    for question, facts in QUESTIONS:
        for f in facts:
            n = texts.get(f["file"], "").count(f["quote"])
            if n != 1:
                out.append(f"{question!r}: the quote occurs {n} times in {f['file']}: {f['quote']!r}")
    return out


def main():
    check = sys.argv[1:] == ["--check"]
    if sys.argv[1:] not in ([], ["--check"]):
        sys.exit(__doc__)
    if not check:
        build(HERE)
    errors = problems()
    if check:
        with tempfile.TemporaryDirectory() as tmp:
            names = build(pathlib.Path(tmp))
            for name in names:
                committed = HERE / name
                if not committed.exists():
                    errors.append(f"{name} is missing")
                elif committed.read_bytes() != (pathlib.Path(tmp) / name).read_bytes():
                    errors.append(f"{name} differs from what make_docs.py writes")
            # Finder writes a .DS_Store into a folder it opens.
            ours = {"make_docs.py", ".gitattributes", ".DS_Store"}
            extra = sorted(p.name for p in HERE.iterdir() if p.name not in ours and p.name not in names)
            errors += [f"{name} is not written by make_docs.py" for name in extra]
    for e in errors:
        print(e, file=sys.stderr)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
